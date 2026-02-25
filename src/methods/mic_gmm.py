
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.mixture import BayesianGaussianMixture

from methods.registry import register_solver
from methods.base_solver import BaseSolver
from models.resnet_attn_fingerprint import resnet50_attn_fingerprint
from utils import AverageMeter, cycle

logger = logging.getLogger(__name__)

class RelationPrototypes:
    """
    1. Mining Inter-Class Relations
    Supports two modes:
    - 'soft_label': Uses Softmax probability distributions and KL Divergence.
    - 'channel': Uses Feature vectors (channels) and Cosine Distance.
    """
    def __init__(self, num_classes, feature_dim=None, mode='soft_label', device='cuda', temperature=2.0):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.mode = mode
        self.device = device
        self.temperature = temperature
        
        # Dimensions depend on mode
        # soft_label: (K, K)
        # channel: (K, D)
        self.prototypes = None 

    def update(self, model, loader):
        """
        Compute inter-class relation prototypes on source data.
        """
        model.eval()
        
        if self.mode == 'soft_label':
            sums = torch.zeros(self.num_classes, self.num_classes, device=self.device)
        else: # channel
            if self.feature_dim is None:
                raise ValueError("feature_dim must be provided for channel mode")
            sums = torch.zeros(self.num_classes, self.feature_dim, device=self.device)
            
        counts = torch.zeros(self.num_classes, device=self.device)
        
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                
                # Get model outputs
                logits, _, attn = model(imgs)
                
                if self.mode == 'soft_label':
                    data = F.softmax(logits, dim=1)
                else:
                    # For channel mode, use Softmax on Projected Features with Temperature
                    data = F.softmax(attn / self.temperature, dim=1)
                
                for c in range(self.num_classes):
                    mask = (labels == c)
                    if mask.sum() > 0:
                        sums[c] += data[mask].sum(dim=0)
                        counts[c] += mask.sum()
        
        # Avoid division by zero
        self.prototypes = sums / (counts.unsqueeze(1) + 1e-8)
        
        # For both modes, prototypes are now distributions -> no extra normalization needed
        # But for stability, ensure they sum to 1 (arithmetic mean of softmax is not necessarily softmax)
        # However, for KL center, mean of probs is a reasonable approximation.
            
        model.train() 

    def get_divergence(self, target_features, target_probs, target_pred_labels):
        """
        2. Known-Unknown Separation
        Computes divergence/distance between sample and its predicted class prototype.
        """
        if self.prototypes is None:
             logger.warning("Prototypes not initialized! Returning 0 divergence.")
             return torch.zeros(target_probs.size(0), device=self.device)
             
        protos = self.prototypes[target_pred_labels] 
        epsilon = 1e-8
        
        if self.mode == 'soft_label':
            # KL Divergence: s_j = D_KL(p_j || tilde{p}_c)
            # KL(P || Q) = sum(P * log(P/Q))
            res = torch.sum(target_probs * (torch.log(target_probs + epsilon) - torch.log(protos + epsilon)), dim=1)
            return res
            
        elif self.mode == 'channel':
            # KL Divergence on Channel Distributions
            # Inputs (target_features) should already be Softmaxed
            
            # Ensure target_features are valid probs (just in case)
            # feats_prob = F.softmax(target_features, dim=1) # Assumed passed as such?
            # Let's assume input is already probabilities or we softmax it here?
            # get_divergence is called with `t_attn`. In train loop we will softmax it.
            
            feats_prob = target_features # Expected to be softmaxed
            
            # KL(P || Q)
            res = torch.sum(feats_prob * (torch.log(feats_prob + epsilon) - torch.log(protos + epsilon)), dim=1)
            return res
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

class BayesianGMMRejection:
    """
    3. Bayesian Gaussian Mixture Model (GMM)
    """
    def __init__(self, queue_size=1024, device='cuda'):
        self.queue_size = queue_size
        self.device = device
        self.score_queue = [] 
        # Dirichlet Process GMM
        self.gmm = BayesianGaussianMixture(
            n_components=10, 
            weight_concentration_prior_type='dirichlet_process',
            random_state=42,
            max_iter=100
        )
        self.fitted = False
        self.known_component_idx = -1
        self.fit_interval = 50 
        self.last_fit_iter = -1

    def push_and_fit(self, scores, current_iter, force=False):
        # Push
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy().tolist()
        
        self.score_queue.extend(scores)
        if len(self.score_queue) > self.queue_size:
            self.score_queue = self.score_queue[-self.queue_size:]
            
        # Fit logic
        if not force and (current_iter - self.last_fit_iter < self.fit_interval):
            return

        if len(self.score_queue) < 100: # Wait for enough data
            return
            
        data = np.array(self.score_queue).reshape(-1, 1)
        
        try:
            # Increase max_iter and n_init for better convergence
            self.gmm.max_iter = 500
            self.gmm.n_init = 2
            self.gmm.fit(data)
            self.fitted = True
            self.last_fit_iter = current_iter
            
            # Identify known component (Cluster with smallest mean Divergence)
            # CRITICAL FIX: Only consider components with non-trivial weight to avoid selecting empty priors
            active_indices = np.where(self.gmm.weights_ > 1e-3)[0]
            if len(active_indices) > 0:
                active_means = self.gmm.means_[active_indices].flatten()
                min_mean_idx = np.argmin(active_means)
                self.known_component_idx = active_indices[min_mean_idx]
            else:
                self.known_component_idx = np.argmin(self.gmm.means_.flatten())

            # Log means for debugging
            # logger.info(f"GMM Fitted. Means: {self.gmm.means_.flatten()}, Weights: {self.gmm.weights_}, Known Idx: {self.known_component_idx}")

        except Exception as e:
            logger.warning(f"GMM fit failed: {e}") 

    def predict_known_prob(self, scores):
        """
        Returns Q(x): 1 if sample belongs to 'Known' (low Divergence) component, 0 otherwise.
        """
        if not self.fitted:
            # Default to 1 (Known) early on to allow adaptation
            return torch.ones(len(scores), device=self.device)

        scores_np = scores.detach().cpu().numpy().reshape(-1, 1)
        predicted_components = self.gmm.predict(scores_np)
        
        # Indicator Q(x)
        is_known = (predicted_components == self.known_component_idx).astype(float)
        return torch.from_numpy(is_known).float().to(self.device)

def random_patch_masking(x, mask_ratio=0.7, patch_size=32):
    """
    MIC Masking Strategy.
    """
    B, C, H, W = x.shape
    h_patches = H // patch_size
    w_patches = W // patch_size
    num_patches = h_patches * w_patches
    num_masked = int(num_patches * mask_ratio)
    
    mask = torch.zeros(B, num_patches, device=x.device)
    noise = torch.rand(B, num_patches, device=x.device)
    _, indices = torch.topk(noise, num_masked, dim=1)
    mask.scatter_(1, indices, 1)
    
    mask = mask.view(B, 1, h_patches, w_patches)
    mask = F.interpolate(mask, size=(H, W), mode='nearest')
    
    return x * (1 - mask)

@register_solver("mic_gmm")
class MICGMMSolver(BaseSolver):
    """
    Improved MIC-GMM Solver with Soft-label Prototypes (or Projected Channel) and DPGMM.
    """
    
    def build_model(self):
        cfg = self.config.method
        
        # Backbone
        self.model = resnet50_attn_fingerprint(
            num_classes=self.num_classes, 
            reduction=cfg.get("se_reduction", 16)
        ).to(self.device)
        
        # Relation Mode (soft_label or channel)
        self.relation_mode = cfg.get("relation_mode", "soft_label")
        logger.info(f"MIC-GMM Relation Mode: {self.relation_mode}")

        # New Modules
        # Feature dim for Attn is 3072
        # Feature dim for Attn is 3072 (1024+2048)
        # User Feedback: Use Layer 3 (1024) only to reduce dimension
        feature_dim = 3072
        self.project_dim = 256
        
        if self.relation_mode == 'channel':
            # Use only Layer 3 features (1024)
            proj_input_dim = 1024 
            self.projector = nn.Linear(proj_input_dim, self.project_dim, bias=False).to(self.device)
            # Initialize projector near identity or orthogonal if possible, but random is fine for start
            nn.init.xavier_uniform_(self.projector.weight)
            # Freeze Projector (Random Projection)
            for p in self.projector.parameters():
                p.requires_grad = False
            target_dim = self.project_dim
        else:
            self.projector = None
            target_dim = feature_dim # Not used for soft_label
        
        self.sl_protos = RelationPrototypes(
            self.num_classes, 
            feature_dim=target_dim, 
            mode=self.relation_mode, 
            device=self.device,
            temperature=0.1 # Low temp for normalized features (cosine-like logits)
        )
        
        self.bgmm = BayesianGMMRejection(
            queue_size=cfg.get("queue_size", 4096),
            device=self.device
        )
        
        # Optimizer
        params = list(self.model.parameters())
        # Projector is frozen, do not add to optimizer
             
        self.optimizer = optim.SGD(
            params, 
            lr=cfg.get("lr", 1e-2), 
            momentum=0.9, 
            weight_decay=1e-3, 
            nesterov=True
        )
        
        # Scheduler
        self.use_scheduler = True
        self.warmup_epochs = 10
        
    def _get_lr_scheduler(self, total_epochs):
        lambda_lr = lambda epoch: 1.0 
        
        if self.use_scheduler:
            def lr_lambda(epoch):
                if epoch < self.warmup_epochs:
                    return float(epoch + 1) / self.warmup_epochs
                else:
                    curr = epoch - self.warmup_epochs
                    total = total_epochs - self.warmup_epochs
                    return 0.5 * (1 + math.cos(math.pi * curr / total))
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        return optim.lr_scheduler.LambdaLR(self.optimizer, lambda_lr)

    def _get_features(self, imgs):
        """Helper to get features, projecting if needed."""
        logits, _, attn = self.model(imgs)
        if self.projector:
            # Channel mode: Use Layer 3 only (first 1024 dims)
            attn_layer3 = attn[:, :1024]
            # Normalize to unit sphere to handle ReLU unboundedness
            attn_layer3 = F.normalize(attn_layer3, p=2, dim=1)
            attn = self.projector(attn_layer3)
        return logits, attn

    def train(self):
        cfg = self.config.method
        max_epochs = cfg.get("epochs", 40)
        start_epoch = 0
        iters_per_epoch = cfg.get("iters_per_epoch", 500)
        
        self.scheduler = self._get_lr_scheduler(max_epochs)
        
        # Iterators
        source_iter = cycle(self.source_loader)
        target_iter = cycle(self.target_loader)
        
        # Params
        lambda_mic = cfg.get("lambda_mic", 0.5)    
        lambda_adapt = cfg.get("lambda_adapt", 0.5) 
        lambda_ent = cfg.get("lambda_ent", 0.1)
        mask_ratio = cfg.get("mic_mask_ratio", 0.7) 
        
        self.model.train()
        if self.projector: self.projector.train()
        
        for epoch in range(start_epoch, max_epochs):
            # 1. Update Relation Prototypes at start of epoch
            # Note: For channel mode, we need to handle projection inside update too?
            # RelationPrototypes.update calls model(imgs). We need to intercept or pass a wrapper.
            # Easiest way: Modify RelationPrototypes.update to accept a projector or handle it here?
            # Better: Pass a custom forward function or modify RelationPrototypes.
            # Let's override RelationPrototypes.update locally for this.
            self._update_prototypes_with_projection()
            
            loss_meters = {
                'total': AverageMeter(), 'cls': AverageMeter(), 
                'mic': AverageMeter(), 'adapt': AverageMeter(), 'ent': AverageMeter(),
                'div': AverageMeter() # Raw Divergence
            }
            
            for i in range(iters_per_epoch):
                current_iter = epoch * iters_per_epoch + i
                
                s_img, s_label = next(source_iter)
                t_img, _ = next(target_iter)
                
                s_img, s_label = s_img.to(self.device), s_label.to(self.device)
                t_img = t_img.to(self.device)
                
                # --- Source Step ---
                s_logits, s_attn = self._get_features(s_img)
                loss_cls = F.cross_entropy(s_logits, s_label)
                
                # --- Target Step ---
                t_logits, t_attn = self._get_features(t_img)
                t_probs = F.softmax(t_logits, dim=1)
                t_preds = torch.argmax(t_probs, dim=1)
                
                # Pre-process for channel mode
                if self.relation_mode == 'channel':
                    # Apply Softmax to projected features with Temperature
                    t_feat_prob = F.softmax(t_attn / 5.0, dim=1)
                    measure_feat = t_feat_prob
                else:
                    measure_feat = t_attn 

                # 2. Compute Divergence
                if self.relation_mode == 'channel':
                    divergence_scores = self.sl_protos.get_divergence(measure_feat, t_probs, t_preds)
                else:
                    divergence_scores = self.sl_protos.get_divergence(t_attn, t_probs, t_preds)
                
                # 3. GMM Fit & Predict
                self.bgmm.push_and_fit(divergence_scores, current_iter)
                weights_known = self.bgmm.predict_known_prob(divergence_scores) # Q(x)
                
                # WARMUP: Force weights_known to 0 initially to allow features to drift naturally
                # before enforcing adaptation or entropy minimization.
                if epoch < 3: 
                    weights_known = torch.zeros_like(weights_known)
                
                # --- MIC Step (Masked Consistency) ---
                t_masked = random_patch_masking(t_img, mask_ratio=mask_ratio)
                _, tm_attn = self._get_features(t_masked)
                loss_mic = F.mse_loss(t_attn, tm_attn)
                
                # --- Adaptation Step ---
                loss_adapt = torch.mean(divergence_scores * weights_known)
                
                # Entropy Minimization (Weighted)
                entropy = -torch.sum(t_probs * torch.log(t_probs + 1e-5), dim=1)
                loss_ent = torch.mean(entropy * weights_known)
                
                # Total Loss
                loss = loss_cls + lambda_mic * loss_mic + \
                       lambda_adapt * loss_adapt + lambda_ent * loss_ent 
                       
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update Meters
                loss_meters['total'].update(loss.item())
                loss_meters['cls'].update(loss_cls.item())
                loss_meters['mic'].update(loss_mic.item())
                loss_meters['adapt'].update(loss_adapt.item())
                loss_meters['ent'].update(loss_ent.item())
                loss_meters['div'].update(divergence_scores.mean().item())
                
                if (i + 1) % 50 == 0:
                    div_min = divergence_scores.min().item()
                    div_max = divergence_scores.max().item()
                    div_std = divergence_scores.std().item()
                    logger.info(f"Epoch [{epoch+1}/{max_epochs}] Iter [{i+1}/{iters_per_epoch}] "
                                f"Loss: {loss_meters['total'].avg:.4f} "
                                f"Div: {loss_meters['div'].avg:.4f} (Min:{div_min:.4f} Max:{div_max:.4f} Std:{div_std:.4f}) "
                                f"ProbKnown: {weights_known.mean().item():.3f}")
                                
            self.scheduler.step()
            
            # Evaluate
            hos = self.evaluate()
            logger.info(f"Epoch {epoch+1} Result - H-score: {hos:.2f}%")
            
            if (epoch + 1) % 5 == 0 or epoch == max_epochs - 1:
                self.save_checkpoint(f"checkpoints/mic_gmm_{self.relation_mode}_epoch_{epoch+1}.pth")

    def _update_prototypes_with_projection(self):
        """Wrapper to update prototypes using projected features if needed."""
        self.model.eval()
        if self.projector: self.projector.eval()
        
        # We need to manually duplicate the logic of RelationPrototypes.update 
        # OR temporarily monkey-patch model?
        # Cleaner: Duplicate the minimal logic here or add `model_forward` callback to RelationPrototypes?
        # Let's create a temporary wrapper model.
        
        class Wrapper(nn.Module):
            def __init__(self, model, projector):
                super().__init__()
                self.model = model
                self.projector = projector
            def forward(self, x):
                l, _, a = self.model(x)
                if self.projector: 
                     # Channel mode: Layer 3 only
                     a = a[:, :1024]
                     # Normalize
                     a = F.normalize(a, p=2, dim=1)
                     a = self.projector(a)
                return l, None, a # Return expected signature (logits, feats, attn)
                
        wrapper = Wrapper(self.model, self.projector)
        self.sl_protos.update(wrapper, self.source_loader)
        
        self.model.train()
        if self.projector: self.projector.train()

    def evaluate(self):
        """Evaluation with DPGMM Rejection."""
        self.model.eval()
        if self.projector: self.projector.eval()
        
        all_preds, all_labels = [], []
        all_known_scores = [] # Q(x)
        
        with torch.no_grad():
            for imgs, labels in self.target_test_loader:
                imgs = imgs.to(self.device)
                logits, attn = self._get_features(imgs)
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                
                # Divergence Scores
                if self.relation_mode == 'channel':
                    feat_probs = F.softmax(attn / 5.0, dim=1)
                    scores = self.sl_protos.get_divergence(feat_probs, probs, preds)
                else:
                    scores = self.sl_protos.get_divergence(attn, probs, preds)
                
                # Predict Known Indication Q(x)
                is_known = self.bgmm.predict_known_prob(scores)
                
                all_preds.append(preds.cpu())
                all_labels.append(labels)
                all_known_scores.append(is_known.cpu())
                
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_known_scores = torch.cat(all_known_scores)
        
        # Apply Rejection
        final_preds = all_preds.clone()
        if self.unknown_label is not None:
            # If Q(x) == 0 -> Unknown
            rejected_mask = all_known_scores < 0.5
            final_preds[rejected_mask] = self.unknown_label
            
        self.model.train()
        if self.projector: self.projector.train()
        
        return self._compute_osda_metrics(final_preds, all_labels)

    def save_checkpoint(self, path):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "method": "mic_gmm",
            "model": self.model.state_dict(),
            "prototypes": self.sl_protos.prototypes,
            "relation_mode": self.relation_mode
        }
        if self.projector:
             state["projector"] = self.projector.state_dict()
             
        torch.save(state, path)
        logger.info(f"Checkpoint saved to {path}")
