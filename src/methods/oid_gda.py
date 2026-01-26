"""
Orthogonal Information Disentanglement for Gated Domain Adaptation (OID-GDA).

Theoretical Principle:
  Input Z is decomposed into Z_inv (Invariant) + Z_sp (Specific) + Z_noise (Noise).
  Objective: Maximize I(Z_inv; Y) while ensuring Z_inv is domain-invariant and Z_sp is domain-specific.

Architecture:
  - Backbone F -> Z
  - Gating G -> w = sigmoid(MLP(Z))
  - Z_inv = Z * w
  - Z_sp = Z * (1 - w)
  - Classifier C -> Y from Z_inv
  - Discriminator D -> d from Z (applied to both Z_inv and Z_sp)

Loss Constraints:
  1. Task: CE(C(Z_inv_s), y_s)
  2. Consistency: || w(x_t) - w(Aug(x_t)) ||^2  [Filter Z_noise]
  3. Orthogonal Decoupling:
     - Adv_Inv: Maximize D error on Z_inv (GRL)
     - Sup_Sp: Minimize D error on Z_sp (Direct)
  4. Entropy: Min H(C(Z_inv_t))
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from tqdm import tqdm

from methods.registry import register_solver
from methods.base_solver import BaseSolver
from models.backbones import get_backbone
from models.heads import ChannelGatingModule, SemanticHead, DomainHead
from utils import AverageMeter, cycle

logger = logging.getLogger(__name__)


class GradReverse(Function):
    """
    Gradient Reversal Layer.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


@register_solver("oid_gda")
class OIDGDASolver(BaseSolver):
    """
    OID-GDA Solver.
    """
    
    def build_model(self):
        """Build F, G, C, D."""
        backbone_name = self.config.method.backbone
        
        # 1. Feature Extractor (F)
        backbone = get_backbone(backbone_name)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.feature_extractor = backbone.to(self.device)
        
        # 2. Gating Network (G)
        gating_hidden = self.config.method.gating_hidden_dim
        self.gating_module = ChannelGatingModule(
            feature_dim=self.feature_dim,
            hidden_dim=gating_hidden
        ).to(self.device)
        
        # 3. Task Classifier (C)
        semantic_hidden = self.config.method.semantic_hidden_dim
        self.classifier = SemanticHead(
            in_features=self.feature_dim,
            num_classes=self.num_classes,
            hidden_dim=semantic_hidden
        ).to(self.device)
        
        # 4. Domain Discriminator (D)
        # Shared discriminator for both streams? 
        # User prompt: "D_dom: Input feature, predict domain label d".
        # We will use one DomainHead.
        discriminator_hidden = self.config.method.discriminator_hidden_dim
        self.discriminator = DomainHead(
            in_features=self.feature_dim,
            hidden_dim=discriminator_hidden
        ).to(self.device)
        
        logger.info(f"Built OID-GDA model components on {self.device}")
        
    def _build_optimizers(self):
        """
        Build optimizers for Alternating Optimization.
        Opt_D: Discriminator
        Opt_G: Generator (Backbone + Gate + Classifier)
        """
        # Opt_D parameters
        self.opt_d = optim.SGD(
            self.discriminator.parameters(),
            lr=self.config.method.lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        
        # Opt_G parameters
        # Note: GRL implementation usually allows single optimizer.
        # But if we want explicit Alternating steps as requested:
        # Step 1: Upd D (Minimize Discrim Error)
        # Step 2: Upd G (Maximize/Minimize depending on term)
        # Using GRL makes Step 2 implicit for Adv part. 
        # But Sup_Sp part is Cooperative (Minimize D error).
        # And Adv_Inv part is Adversarial (Maximize D error).
        # We can put everything in one optimizer using GRL for Adv_Inv 
        # and standard grad for Sup_Sp.
        # BUT, standard DANN usually updates D k times then G 1 time.
        # Let's use a single optimizer first for simplicity, leveraging GRL.
        # "Alternating Optimization" typically implies distinct backward passes.
        # Let's stick to standard single-opt DANN style which effectively does this simultaneously,
        # UNLESS explicit steps are strictly required. 
        # Prompt: "Training Strategy: Alternating Optimization. Step 1 D, Step 2 G".
        # Okay, I will implement explicit alternating steps to be safe.
        
        g_params = (
            list(self.feature_extractor.parameters()) + 
            list(self.gating_module.parameters()) +
            list(self.classifier.parameters())
        )
        self.opt_g = optim.SGD(
            g_params,
            lr=self.config.method.lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        
    def _set_train_mode(self):
        self.feature_extractor.train()
        self.gating_module.train()
        self.classifier.train()
        self.discriminator.train()
        
    def _forward_decompose(self, x):
        f = self.feature_extractor(x)
        gate = self.gating_module(f)
        f_inv = f * gate
        f_sp = f * (1.0 - gate)
        return f, f_inv, f_sp, gate

    def train(self):
        self._build_optimizers()
        
        pretrain_epochs = self.config.method.pretrain_epochs
        adapt_epochs = self.config.method.adapt_epochs
        
        if pretrain_epochs > 0:
            logger.info("Stage 1: Pretraining Source...")
            self._train_pretrain(pretrain_epochs)
            
        logger.info("Stage 2: OID-GDA Adaptation...")
        self._train_adapt(adapt_epochs)
        
    def _train_pretrain(self, epochs):
        # Standard Source Training
        for epoch in range(epochs):
            self._set_train_mode()
            meter = AverageMeter()
            pbar = tqdm(self.source_loader, desc=f"Pretrain {epoch+1}/{epochs}")
            
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                
                self.opt_g.zero_grad()
                _, f_inv, _, _ = self._forward_decompose(imgs)
                pred = self.classifier(f_inv)
                loss = self.criterion(pred, labels)
                loss.backward()
                self.opt_g.step()
                
                meter.update(loss.item())
                pbar.set_postfix({"loss": meter.avg})
                
    def _train_adapt(self, epochs):
        """
        Alternating Optimization Loop.
        Step 1: Update D (Discrim loss on Inv and Sp).
        Step 2: Update G (Task loss, Adv on Inv, Coop on Sp, Consist, Ent).
        """
        for epoch in range(epochs):
            self._set_train_mode()
            tgt_iter = cycle(self.target_loader)
            
            meters = {k: AverageMeter() for k in ["cls", "d_inv", "d_sp", "adv", "sup", "con", "ent"]}
            
            pbar = tqdm(self.source_loader, desc=f"Adapt {epoch+1}/{epochs}")
            
            for src_imgs, src_labels in pbar:
                tgt_imgs, _ = next(tgt_iter)
                # Strong aug for consistency
                # Simulate strong aug by flipping or noise if no aug loader
                tgt_imgs_aug = torch.flip(tgt_imgs, dims=[3]) 
                
                src_imgs, src_labels = src_imgs.to(self.device), src_labels.to(self.device)
                tgt_imgs = tgt_imgs.to(self.device)
                tgt_imgs_aug = tgt_imgs_aug.to(self.device)
                
                # --- Step 1: Update Discriminator ---
                # D needs to distinguish Source/Target for BOTH Z_inv and Z_sp.
                # Maximize D accuracy (Minimize BCE).
                
                # Forward (Detached G)
                with torch.no_grad():
                     _, f_inv_s, f_sp_s, _ = self._forward_decompose(src_imgs)
                     _, f_inv_t, f_sp_t, _ = self._forward_decompose(tgt_imgs)
                
                self.opt_d.zero_grad()
                
                # D on Inv
                d_inv_s = self.discriminator(f_inv_s.detach())
                d_inv_t = self.discriminator(f_inv_t.detach())
                loss_d_inv = (
                    F.binary_cross_entropy_with_logits(d_inv_s, torch.ones_like(d_inv_s)) + 
                    F.binary_cross_entropy_with_logits(d_inv_t, torch.zeros_like(d_inv_t))
                ) * 0.5
                
                # D on Sp
                d_sp_s = self.discriminator(f_sp_s.detach())
                d_sp_t = self.discriminator(f_sp_t.detach())
                loss_d_sp = (
                    F.binary_cross_entropy_with_logits(d_sp_s, torch.ones_like(d_sp_s)) + 
                    F.binary_cross_entropy_with_logits(d_sp_t, torch.zeros_like(d_sp_t))
                ) * 0.5
                
                loss_d_total = loss_d_inv + loss_d_sp
                loss_d_total.backward()
                self.opt_d.step()
                
                # --- Step 2: Update Generator (F, G, C) ---
                self.opt_g.zero_grad()
                
                # Re-forward (Attached G)
                _, f_inv_s, f_sp_s, gate_s = self._forward_decompose(src_imgs)
                _, f_inv_t, f_sp_t, gate_t = self._forward_decompose(tgt_imgs)
                _, _, _, gate_t_aug = self._forward_decompose(tgt_imgs_aug)
                
                # 1. Task Loss
                pred_s = self.classifier(f_inv_s)
                loss_task = self.criterion(pred_s, src_labels)
                
                # 2. Consistency Loss (on Gate)
                # ||w(x) - w(x')||^2
                loss_consist = F.mse_loss(gate_t, gate_t_aug)
                
                # 3. Orthogonal Decoupling
                # A. Adv Inv: Fool D (Maximize domain confusion)
                # We can use GRL or just invert labels since we do alternating steps.
                # Standard Generator step in GAN: Maximize log(D(G(z))).
                # Or invert labels: Labels -> Target=1, Source=0.
                # Let's use Inverted Labels for explicit step.
                # Z_inv_s should be classified as Target (0) -> No, confusion means 0.5.
                # Standard "Fool D": Minimize BCE(D(z_s), 0) -> Minimize BCE(D(z_s), 1)??
                # GRL way: Minimize BCE(D(z_s), 1) (Correct label) but with neg grad.
                # Explicit way: Minimize BCE(D(z_s), 0) (Wrong label).
                # Let's use Inverted Labels strategy.
                d_pred_inv_s = self.discriminator(f_inv_s)
                d_pred_inv_t = self.discriminator(f_inv_t)
                loss_adv_inv = (
                    F.binary_cross_entropy_with_logits(d_pred_inv_s, torch.zeros_like(d_pred_inv_s)) +
                    F.binary_cross_entropy_with_logits(d_pred_inv_t, torch.ones_like(d_pred_inv_t))
                ) * 0.5
                
                # B. Sup Sp: Help D (Cooperative)
                # Minimize BCE(D(z_s), 1) (Correct label)
                d_pred_sp_s = self.discriminator(f_sp_s)
                d_pred_sp_t = self.discriminator(f_sp_t)
                loss_sup_sp = (
                    F.binary_cross_entropy_with_logits(d_pred_sp_s, torch.ones_like(d_pred_sp_s)) +
                    F.binary_cross_entropy_with_logits(d_pred_sp_t, torch.zeros_like(d_pred_sp_t))
                ) * 0.5
                
                # 4. Entropy
                pred_t = self.classifier(f_inv_t)
                prob_t = F.softmax(pred_t, dim=1)
                loss_ent = -(prob_t * torch.log(pred_t.softmax(1) + 1e-6)).sum(1).mean()
                
                # Total
                cfg = self.config.method
                loss_g_total = (
                    loss_task + 
                    cfg.lambda_consist * loss_consist +
                    cfg.lambda_adv_inv * loss_adv_inv + 
                    cfg.lambda_sup_sp * loss_sup_sp +
                    cfg.lambda_ent * loss_ent
                )
                
                loss_g_total.backward()
                self.opt_g.step()
                
                # Logging
                meters["cls"].update(loss_task.item())
                meters["d_inv"].update(loss_d_inv.item())
                meters["d_sp"].update(loss_d_sp.item())
                meters["adv"].update(loss_adv_inv.item())
                meters["sup"].update(loss_sup_sp.item())
                meters["con"].update(loss_consist.item())
                meters["ent"].update(loss_ent.item())
                
                pbar.set_postfix({
                    "cls": f"{meters['cls'].avg:.3f}",
                    "adv": f"{meters['adv'].avg:.3f}",
                    "con": f"{meters['con'].avg:.3f}"
                })
                
            # Eval
            acc = self.evaluate()
            logger.info(f"Adapt Epoch {epoch+1} Acc: {acc:.2f}%")

    def forward_for_eval(self, imgs):
        _, f_inv, _, _ = self._forward_decompose(imgs)
        return self.classifier(f_inv)

    def save_checkpoint(self, path):
        torch.save({
            "method": "oid_gda",
            "feature_extractor": self.feature_extractor.state_dict(),
            "gating_module": self.gating_module.state_dict(),
            "classifier": self.classifier.state_dict(),
            "discriminator": self.discriminator.state_dict(),
        }, path)

    def load_checkpoint(self, path):
         checkpoint = torch.load(path, map_location=self.device)
         self.feature_extractor.load_state_dict(checkpoint["feature_extractor"])
         self.gating_module.load_state_dict(checkpoint["gating_module"])
         self.classifier.load_state_dict(checkpoint["classifier"])
         self.discriminator.load_state_dict(checkpoint["discriminator"])
