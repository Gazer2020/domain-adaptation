"""
Gated Adaptation Domain (GAD) for Open Set Domain Adaptation.

Combines best practices from TOD and CAD:
- Dual-path decomposition: f_inv = f × gate, f_sp = f × (1-gate)
- Unified gate loss: sparsity + class consistency
- MMD alignment on invariant features
- Pseudo-labeling for target domain
- GMM-based adaptive rejection threshold

Target: H-score 85+
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import numpy as np

from methods.registry import register_solver
from methods.base_solver import BaseSolver
from models.backbones import get_backbone
from models.heads import SemanticHead
from utils import AverageMeter, cycle


logger = logging.getLogger(__name__)


class GatingModule(nn.Module):
    """
    Channel Gating with Temperature Control.
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or feature_dim // 4
        
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, features: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Compute gate values with temperature scaling."""
        logits = self.net(features)
        return torch.sigmoid(logits / temperature)


def compute_mmd(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute MMD with RBF kernel."""
    if source.size(0) == 0 or target.size(0) == 0:
        return torch.tensor(0.0, device=source.device)
    
    with torch.no_grad():
        all_data = torch.cat([source, target], dim=0)
        pairwise_dist = torch.cdist(all_data, all_data, p=2)
        sigma = torch.median(pairwise_dist[pairwise_dist > 0]).clamp(min=0.1).item()
    
    def rbf(x, y):
        return torch.exp(-torch.cdist(x, y, p=2) ** 2 / (2 * sigma ** 2))
    
    k_ss, k_tt, k_st = rbf(source, source), rbf(target, target), rbf(source, target)
    return (k_ss.mean() + k_tt.mean() - 2 * k_st.mean()).clamp(min=0.0)


@register_solver("gad")
class GADSolver(BaseSolver):
    """
    GAD Solver: Gated Adaptation Domain.
    
    Simplified but effective approach:
    - Dual-path decomposition (f*gate, f*(1-gate))
    - Gate regularization with sparsity + class consistency
    - MMD alignment + strong pseudo-labeling
    """

    def build_model(self):
        """Build backbone, gating module, and classifier."""
        backbone = get_backbone(self.config.method.backbone)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone.to(self.device)
        
        self.gating = GatingModule(
            self.feature_dim, 
            self.config.method.gating_hidden_dim
        ).to(self.device)
        
        self.classifier = SemanticHead(
            self.feature_dim, 
            self.num_classes,
            self.config.method.semantic_hidden_dim
        ).to(self.device)
        
        self.num_src_classes = self.num_classes - 1 if self.unknown_label else self.num_classes
        self.prototypes = torch.zeros(self.num_src_classes, self.feature_dim, device=self.device)
        self.threshold = 0.5
        self.temperature = self.config.method.init_temperature
        
        logger.info(f"Built GAD model: backbone={self.config.method.backbone}, "
                    f"feat_dim={self.feature_dim}, classes={self.num_classes}")

    def _get_trainable_params(self):
        return list(self.backbone.parameters()) + \
               list(self.gating.parameters()) + \
               list(self.classifier.parameters())

    def _build_optimizer(self):
        self.optimizer = optim.SGD(
            self._get_trainable_params(),
            lr=self.config.method.lr,
            momentum=0.9, weight_decay=5e-4, nesterov=True
        )

    def _set_train_mode(self):
        self.backbone.train()
        self.gating.train()
        self.classifier.train()

    def _set_eval_mode(self):
        self.backbone.eval()
        self.gating.eval()
        self.classifier.eval()

    def _decompose(self, x: torch.Tensor, temp: float = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Core decomposition: f_inv = f * gate, f_sp = f * (1-gate)"""
        temp = temp or self.temperature
        f = self.backbone(x)
        gate = self.gating(f, temp)
        return f * gate, f * (1 - gate), gate

    def _compute_gate_loss(self, gates: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        Unified gate regularization:
        1. Sparsity: binary entropy pushes gates to 0/1
        2. Consistency: same class → similar gates
        """
        eps = 1e-6
        # Binary entropy for sparsity
        entropy = -gates * torch.log(gates + eps) - (1 - gates) * torch.log(1 - gates + eps)
        loss = entropy.mean()
        
        if labels is not None:
            # Class consistency
            centers = torch.zeros(self.num_src_classes, self.feature_dim, device=self.device)
            counts = torch.zeros(self.num_src_classes, device=self.device)
            centers.index_add_(0, labels, gates)
            counts.index_add_(0, labels, torch.ones(len(labels), device=self.device))
            centers = centers / counts.clamp(min=1).unsqueeze(1)
            loss = loss + F.mse_loss(gates, centers[labels].detach())
        
        return loss

    def _get_temperature(self, epoch: int, total: int) -> float:
        """Linear temperature decay."""
        cfg = self.config.method
        progress = epoch / max(total - 1, 1)
        return cfg.init_temperature + (cfg.final_temperature - cfg.init_temperature) * progress

    def train(self):
        """Training loop."""
        self._build_optimizer()
        cfg = self.config.method
        warmup_epochs = cfg.pretrain_epochs
        adapt_epochs = cfg.adapt_epochs
        total_epochs = warmup_epochs + adapt_epochs
        
        logger.info(f"Training GAD: {warmup_epochs} warmup + {adapt_epochs} adapt epochs")
        
        for epoch in range(total_epochs):
            is_warmup = epoch < warmup_epochs
            self.temperature = cfg.init_temperature if is_warmup else \
                               self._get_temperature(epoch - warmup_epochs, adapt_epochs)
            
            metrics = self._train_epoch(epoch, is_warmup)
            
            if not is_warmup or epoch == warmup_epochs - 1:
                self._compute_prototypes()
                h_score = self.evaluate()
                metrics['h_score'] = h_score
            
            self._log_epoch(epoch, total_epochs, is_warmup, metrics)
        
        self._compute_prototypes()
        h_score = self.evaluate()
        logger.info(f"Training complete. Final H-score: {h_score:.2f}%")

    def _train_epoch(self, epoch: int, is_warmup: bool) -> dict:
        """Single epoch training."""
        self._set_train_mode()
        cfg = self.config.method
        
        meters = {k: AverageMeter() for k in ['loss', 'cls', 'align', 'gate', 'gate_mean']}
        tgt_iter = None if is_warmup else cycle(self.target_loader)
        
        pbar = tqdm(self.source_loader, desc=f"Epoch {epoch+1}", leave=False)
        for src_imgs, src_labels in pbar:
            src_imgs, src_labels = src_imgs.to(self.device), src_labels.to(self.device)
            self.optimizer.zero_grad()
            
            # Source forward
            f_inv_s, f_sp_s, gate_s = self._decompose(src_imgs, self.temperature)
            logits_s = self.classifier(f_inv_s)
            loss_cls = self.criterion(logits_s, src_labels)
            loss_gate = self._compute_gate_loss(gate_s, src_labels)
            
            loss_align = torch.tensor(0.0, device=self.device)
            
            if not is_warmup:
                # Target forward
                tgt_imgs, _ = next(tgt_iter)
                tgt_imgs = tgt_imgs.to(self.device)
                f_inv_t, f_sp_t, gate_t = self._decompose(tgt_imgs, self.temperature)
                
                # MMD alignment
                loss_align = compute_mmd(f_inv_s, f_inv_t)
                
                # Pseudo-labeling
                with torch.no_grad():
                    probs_t = F.softmax(self.classifier(f_inv_t), dim=1)
                    conf, pseudo = probs_t.max(dim=1)
                    mask = (conf >= cfg.pseudo_threshold) & (pseudo < self.num_src_classes)
                
                if mask.sum() > 0:
                    loss_cls = loss_cls + cfg.lambda_pseudo * self.criterion(
                        self.classifier(f_inv_t)[mask], pseudo[mask]
                    )
                    loss_gate = loss_gate + self._compute_gate_loss(gate_t[mask], pseudo[mask])
                else:
                    loss_gate = loss_gate + self._compute_gate_loss(gate_t)
            
            # Total loss
            loss = loss_cls + cfg.lambda_align * loss_align + cfg.lambda_gate * loss_gate
            loss.backward()
            self.optimizer.step()
            
            meters['loss'].update(loss.item())
            meters['cls'].update(loss_cls.item())
            meters['align'].update(loss_align.item())
            meters['gate'].update(loss_gate.item())
            meters['gate_mean'].update(gate_s.mean().item())
            
            pbar.set_postfix(loss=f"{meters['loss'].avg:.3f}", gate=f"{meters['gate_mean'].avg:.3f}")
        
        return {k: m.avg for k, m in meters.items()}

    def _log_epoch(self, epoch: int, total: int, is_warmup: bool, metrics: dict):
        """Epoch logging."""
        stage = "Warmup" if is_warmup else "Adapt"
        h_score_str = f", H-score={metrics['h_score']:.2f}%" if 'h_score' in metrics else ""
        logger.info(f"{stage} Epoch {epoch+1}/{total}: loss={metrics['loss']:.4f}, "
                    f"cls={metrics['cls']:.4f}, align={metrics['align']:.4f}, "
                    f"gate_μ={metrics['gate_mean']:.3f}{h_score_str}")

    def _compute_prototypes(self):
        """Compute class prototypes and GMM threshold."""
        self._set_eval_mode()
        
        sums = torch.zeros(self.num_src_classes, self.feature_dim, device=self.device)
        counts = torch.zeros(self.num_src_classes, device=self.device)
        
        with torch.no_grad():
            for imgs, labels in self.source_loader:
                f_inv, _, _ = self._decompose(imgs.to(self.device))
                sums.index_add_(0, labels.to(self.device), f_inv)
                counts.index_add_(0, labels.to(self.device), torch.ones(len(labels), device=self.device))
        
        self.prototypes = sums / counts.clamp(min=1).unsqueeze(1)
        
        dists = []
        with torch.no_grad():
            for imgs, _ in self.target_loader:
                f_inv, _, _ = self._decompose(imgs.to(self.device))
                d = torch.cdist(f_inv, self.prototypes, p=2).min(dim=1)[0]
                dists.append(d.cpu())
        
        dists = torch.cat(dists).numpy().reshape(-1, 1)
        if len(dists) >= 10:
            try:
                gmm = GaussianMixture(n_components=2, covariance_type='spherical', 
                                      reg_covar=1e-3, random_state=42)
                gmm.fit(dists)
                means, vars_ = gmm.means_.flatten(), gmm.covariances_.flatten()
                
                unk_idx = 0 if means[0] > means[1] else 1
                knw_idx = 1 - unk_idx
                
                self.threshold = float((means[unk_idx] * vars_[knw_idx] + means[knw_idx] * vars_[unk_idx]) / 
                                       (vars_[unk_idx] + vars_[knw_idx]))
                logger.info(f"GMM threshold: {self.threshold:.4f} "
                            f"(known_μ={means[knw_idx]:.4f}, unknown_μ={means[unk_idx]:.4f})")
            except Exception as e:
                self.threshold = float(np.median(dists))
                logger.warning(f"GMM failed, using median: {self.threshold:.4f}")
        else:
            self.threshold = float(np.median(dists)) if len(dists) > 0 else 10.0

    def evaluate(self) -> float:
        """Evaluate with prototype-based rejection."""
        if self.prototypes.abs().sum() == 0:
            self._compute_prototypes()
        
        self._set_eval_mode()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for imgs, labels in self.target_test_loader:
                f_inv, _, _ = self._decompose(imgs.to(self.device))
                preds = self.classifier(f_inv).argmax(dim=1)
                
                dists = torch.cdist(f_inv, self.prototypes, p=2).min(dim=1)[0]
                if self.unknown_label is not None:
                    preds[dists > self.threshold] = self.unknown_label
                
                all_preds.append(preds.cpu())
                all_labels.append(labels)
        
        all_preds, all_labels = torch.cat(all_preds), torch.cat(all_labels)
        
        if self.unknown_label is not None and self.setting in ["osda", "unida"]:
            return self._compute_osda_metrics(all_preds, all_labels)
        else:
            return 100 * (all_preds == all_labels).sum().item() / len(all_labels)

    def forward_for_eval(self, imgs):
        """Forward pass for evaluation."""
        f_inv, _, _ = self._decompose(imgs)
        return self.classifier(f_inv)

    def save_checkpoint(self, path):
        torch.save({
            "method": "gad",
            "backbone": self.backbone.state_dict(),
            "gating": self.gating.state_dict(),
            "classifier": self.classifier.state_dict(),
            "prototypes": self.prototypes,
            "threshold": self.threshold,
        }, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.backbone.load_state_dict(ckpt["backbone"])
        self.gating.load_state_dict(ckpt["gating"])
        self.classifier.load_state_dict(ckpt["classifier"])
        if "prototypes" in ckpt:
            self.prototypes = ckpt["prototypes"].to(self.device)
        if "threshold" in ckpt:
            self.threshold = ckpt["threshold"]
        logger.info(f"Loaded checkpoint: {path}")
