import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from methods.base_solver import BaseSolver
from methods.registry import register_solver
from models.backbones import get_backbone

logger = logging.getLogger(__name__)

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)

class DomainClassifier(nn.Module):
    def __init__(self, in_features, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.fc1(x)
        z_domain = self.relu(features)
        logits = self.fc2(z_domain)
        return logits, z_domain

class DomainModulation(nn.Module):
    def __init__(self, in_features, domain_dim=512):
        super().__init__()
        self.fc_gamma = nn.Linear(domain_dim, in_features)
        self.fc_beta = nn.Linear(domain_dim, in_features)

        # Initialize to identity modulation: gamma=0, beta=0
        nn.init.zeros_(self.fc_gamma.weight)
        nn.init.zeros_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, x, z_domain):
        # We add 1.0 to gamma so that when it's 0, modulation is identity
        gamma = self.fc_gamma(z_domain)
        beta = self.fc_beta(z_domain)
        # Residual connection style modulation
        return x * (1.0 + gamma) + beta


class DCFMNetwork(nn.Module):
    def __init__(self, backbone_name, num_classes):
        super().__init__()
        self.backbone = get_backbone(backbone_name)
        
        if hasattr(self.backbone, 'fc'):
            self.feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise NotImplementedError("Backbone representation feature dimension not found.")
            
        self.domain_classifier = DomainClassifier(self.feat_dim)
        self.modulator = DomainModulation(self.feat_dim)
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x, alpha=1.0):
        features = self.backbone(x)
        # GradientReversal forces the backbone to be domain-invariant, acting like DANN.
        # Meanwhile, z_domain captures the remaining unaligned domain specific signals for modulation.
        rev_features = grad_reverse(features, alpha)
        domain_logits, z_domain = self.domain_classifier(rev_features)
        mod_features = self.modulator(features, z_domain)
        task_logits = self.classifier(mod_features)
        return task_logits, domain_logits, mod_features


@register_solver("dcfm")
class DCFMSolver(BaseSolver):
    """
    Domain-Conditioned Feature Modulation (DCFM) solver.
    """
    def build_model(self):
        backbone_name = self.config.method.get("backbone", "resnet50")
        self.net = DCFMNetwork(backbone_name, self.num_classes).to(self.device)

        self.lambda_domain = self.config.method.get("lambda_domain", 1.0)
        self.lambda_target = self.config.method.get("lambda_target", 1.0)
        self.confidence_threshold = self.config.method.get("confidence_threshold", 0.9)

        self.criterion_task = nn.CrossEntropyLoss()
        self.criterion_domain = nn.BCEWithLogitsLoss()

    def forward_for_eval(self, imgs):
        task_logits, _, _ = self.net(imgs)
        return task_logits

    def train(self):
        import torch.optim as optim
        from tqdm import tqdm
        from utils import AverageMeter
        
        max_epochs = self.config.method.epochs
        lr = self.config.method.lr
        
        import math
        optimizer = optim.SGD([
            {'params': self.net.backbone.parameters(), 'lr': lr * 0.1},
            {'params': self.net.domain_classifier.parameters(), 'lr': lr},
            {'params': self.net.modulator.parameters(), 'lr': lr},
            {'params': self.net.classifier.parameters(), 'lr': lr}
        ], momentum=0.9, weight_decay=5e-4)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs * len(self.source_loader))

        logger.info(f"Start training DCFM for {max_epochs} epochs...")

        best_acc = 0.0
        
        for epoch in range(max_epochs):
            self.net.train()
            loss_task_meter = AverageMeter()
            loss_domain_meter = AverageMeter()
            loss_target_meter = AverageMeter()
            loss_total_meter = AverageMeter()

            # Iterate over batches from both source and target domains
            target_iter = iter(self.target_loader)
            
            pbar = tqdm(self.source_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
            for step, (src_imgs, src_labels) in enumerate(pbar):
                try:
                    tgt_imgs, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(self.target_loader)
                    tgt_imgs, _ = next(target_iter)

                src_imgs = src_imgs.to(self.device)
                src_labels = src_labels.to(self.device)
                tgt_imgs = tgt_imgs.to(self.device)

                batch_size_src = src_imgs.size(0)
                batch_size_tgt = tgt_imgs.size(0)

                # Concatenate source and target images
                all_imgs = torch.cat([src_imgs, tgt_imgs], dim=0)

                # Alpha for Gradient Reversal Layer (standard DANN schedule)
                p_alpha = float(epoch * len(self.source_loader) + step) / (max_epochs * len(self.source_loader))
                alpha = 2. / (1. + math.exp(-10 * p_alpha)) - 1
                
                # Forward pass
                optimizer.zero_grad()
                task_logits, domain_logits, _ = self.net(all_imgs, alpha=alpha)

                # Separate source and target logits
                task_logits_src = task_logits[:batch_size_src]
                task_logits_tgt = task_logits[batch_size_src:]
                domain_logits_src = domain_logits[:batch_size_src]
                domain_logits_tgt = domain_logits[batch_size_src:]

                # 1. Task Loss (Source only)
                loss_task = self.criterion_task(task_logits_src, src_labels)

                # 2. Domain Loss (Source and Target)
                # target label for source=0, target=1
                domain_labels_src = torch.zeros(batch_size_src, 1).to(self.device)
                domain_labels_tgt = torch.ones(batch_size_tgt, 1).to(self.device)
                
                loss_domain_src = self.criterion_domain(domain_logits_src, domain_labels_src)
                loss_domain_tgt = self.criterion_domain(domain_logits_tgt, domain_labels_tgt)
                loss_domain = (loss_domain_src + loss_domain_tgt) / 2.0

                # 3. Target Pseudo-label Loss
                probs_tgt = torch.softmax(task_logits_tgt.detach(), dim=1)
                max_probs, pseudo_labels = torch.max(probs_tgt, dim=1)
                
                mask = max_probs >= self.confidence_threshold
                if mask.sum() > 0:
                    loss_target = self.criterion_task(task_logits_tgt[mask], pseudo_labels[mask])
                else:
                    loss_target = torch.tensor(0.0).to(self.device)

                # Warmup for target loss to prevent early noisy pseudo-labels from destroying training
                p = float(epoch * len(self.source_loader) + step) / (max_epochs * len(self.source_loader))
                current_lambda_target = self.lambda_target * (2. / (1. + math.exp(-10 * p)) - 1)

                # Total Loss
                loss_total = loss_task + (self.lambda_domain * loss_domain) + (current_lambda_target * loss_target)

                loss_total.backward()
                optimizer.step()
                scheduler.step()
                
                loss_task_meter.update(loss_task.item())
                loss_domain_meter.update(loss_domain.item())
                loss_target_meter.update(loss_target.item())
                loss_total_meter.update(loss_total.item())

                pbar.set_postfix({
                    "task": f"{loss_task_meter.avg:.3f}",
                    "dom": f"{loss_domain_meter.avg:.3f}",
                    "tgt": f"{loss_target_meter.avg:.3f}",
                    "tot": f"{loss_total_meter.avg:.3f}"
                })
            
            acc = self.evaluate()
            logger.info(f"Epoch {epoch+1} finished. Total Loss: {loss_total_meter.avg:.4f}, Acc: {acc:.2f}%")
            
            if acc > best_acc:
                best_acc = acc
                
        logger.info(f"Training finished. Best Acc: {best_acc:.2f}%")
