"""
ODCFM (OSDA) from scratch: Domain-Conditioned Open-Set.

Core idea: accept domain information and use it twice:
- Feature path: FiLM(h, z_domain) for domain-conditioned representation.
- Open-set path: domain-conditioned score calibration d_dc = a(z_domain)*d + b(z_domain),
  where d is distance-to-source-prototypes. A target-only 2-GMM on d_dc gives a
  dynamic threshold and an unknown posterior used inside L_tgt_unified.

Reject-first variant: network outputs K+1 logits (keeps framework shape), but open-set
decisions are made via d_dc thresholding; logits for the last (unknown) class are
not relied upon for rejection.
"""

import logging
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans

from methods.base_solver import BaseSolver
from methods.registry import register_solver
from models.backbones import get_backbone
from utils import AverageMeter, cycle

logger = logging.getLogger(__name__)


def _normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True).clamp_min(eps))

def supervised_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    tau: float = 0.1,
) -> torch.Tensor:
    """
    Supervised NT-Xent / SupCon for a single-view batch.
    Samples without positives in the batch are ignored.
    """
    if z.ndim != 2:
        raise ValueError("z must be [N, D]")
    labels = labels.view(-1)
    N = z.size(0)
    if N <= 1:
        return torch.zeros((), device=z.device, dtype=z.dtype)

    z = _normalize(z, dim=1)
    sim = (z @ z.t()) / float(tau)
    eye = torch.eye(N, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, float("-inf"))

    pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~eye)
    pos_count = pos.sum(dim=1)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    mean_log_prob_pos = (log_prob.masked_fill(~pos, 0.0).sum(dim=1) / pos_count.clamp_min(1))
    valid = pos_count > 0
    if not valid.any():
        return torch.zeros((), device=z.device, dtype=z.dtype)
    return (-mean_log_prob_pos[valid]).mean()


def cosine_distance_to_prototypes(z: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
    """
    z: [B, D]
    protos: [K, D]
    returns d: [B], where d = 1 - max_k cos(z, proto_k)
    """
    z = _normalize(z, dim=1)
    protos = _normalize(protos, dim=1)
    sim = z @ protos.t()
    max_sim, _ = sim.max(dim=1)
    return 1.0 - max_sim


def _gmm_intersection_1d(
    mean_a: float,
    var_a: float,
    weight_a: float,
    mean_b: float,
    var_b: float,
    weight_b: float,
) -> float:
    """
    Solve for x where weight_a * N(x|mean_a,var_a) == weight_b * N(x|mean_b,var_b).
    """
    var_a = max(float(var_a), 1e-12)
    var_b = max(float(var_b), 1e-12)
    weight_a = max(float(weight_a), 1e-12)
    weight_b = max(float(weight_b), 1e-12)

    A = 0.5 * (1.0 / var_a - 1.0 / var_b)
    B = (mean_b / var_b) - (mean_a / var_a)
    C = (
        0.5 * (mean_a * mean_a / var_a - mean_b * mean_b / var_b)
        + 0.5 * math.log(var_a / var_b)
        + math.log(weight_b / weight_a)
    )

    if abs(A) < 1e-12:
        if abs(B) < 1e-12:
            return 0.5 * (mean_a + mean_b)
        return -C / B

    disc = B * B - 4.0 * A * C
    if disc < 0:
        return 0.5 * (mean_a + mean_b)

    sqrt_disc = math.sqrt(disc)
    r1 = (-B + sqrt_disc) / (2.0 * A)
    r2 = (-B - sqrt_disc) / (2.0 * A)
    lo, hi = (mean_a, mean_b) if mean_a <= mean_b else (mean_b, mean_a)
    between = [r for r in (r1, r2) if lo <= r <= hi]
    if between:
        mid = 0.5 * (lo + hi)
        return min(between, key=lambda r: abs(r - mid))
    mid = 0.5 * (lo + hi)
    return r1 if abs(r1 - mid) < abs(r2 - mid) else r2


def _gaussian_pdf(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    var = var.clamp_min(1e-12)
    return torch.exp(-0.5 * (x - mean) ** 2 / var) / torch.sqrt(2.0 * math.pi * var)


class DomainEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_domain = F.relu(self.ln1(self.fc1(h)))
        dom_logits = self.fc2(z_domain)
        return dom_logits, z_domain


class FiLM(nn.Module):
    def __init__(self, feat_dim: int, domain_dim: int, scale: float = 4.0):
        super().__init__()
        self.scale = float(scale)
        self.gamma = nn.Sequential(nn.Linear(domain_dim, feat_dim), nn.Sigmoid())
        self.beta = nn.Linear(domain_dim, feat_dim)

        nn.init.zeros_(self.gamma[0].weight)
        nn.init.constant_(self.gamma[0].bias, -math.log(self.scale - 1.0))
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, h_normed: torch.Tensor, z_domain: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(z_domain) * self.scale
        beta = self.beta(z_domain)
        return gamma * h_normed + beta


class ScoreCalibrator(nn.Module):
    """
    Domain-conditioned 1D score calibrator: d_dc = a(z_domain)*d + b(z_domain).
    a(z_domain) is constrained positive via softplus; b(z_domain) is unconstrained.
    """

    def __init__(self, domain_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(domain_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, d: torch.Tensor, z_domain: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ab = self.mlp(z_domain)  # [B,2]
        a_raw, b = ab[:, 0], ab[:, 1]
        a = F.softplus(a_raw) + 1e-3
        d_dc = a * d + b
        return d_dc, a, b


class ODCFMNet(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        domain_hidden_dim: int = 256,
        bottleneck_dim: int = 0,
        film_scale: float = 4.0,
        score_hidden_dim: int = 64,
    ):
        super().__init__()
        self.backbone = get_backbone(backbone_name)
        if not hasattr(self.backbone, "fc"):
            raise NotImplementedError("Backbone feature dimension not found (expected .fc).")
        self.feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.feat_bn = nn.BatchNorm1d(self.feat_dim)
        self.domain = DomainEncoder(self.feat_dim, domain_hidden_dim)
        self.film = FiLM(self.feat_dim, domain_hidden_dim, film_scale)
        self.score = ScoreCalibrator(domain_hidden_dim, score_hidden_dim)

        if bottleneck_dim > 0:
            self.classifier = nn.Sequential(
                nn.Linear(self.feat_dim, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(bottleneck_dim, num_classes),
            )
        else:
            self.classifier = nn.Linear(self.feat_dim, num_classes)

    def extract_h(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.extract_h(x)
        dom_logits_detached, _ = self.domain(h.detach())
        _, z_domain = self.domain(h)
        z = self.feat_bn(h)
        z = self.film(z, z_domain)
        logits = self.classifier(z)
        return logits, dom_logits_detached


@register_solver("odcfm")
class ODCFMSolver(BaseSolver):
    def build_model(self):
        backbone_name = self.config.method.get("backbone", "resnet50")
        domain_hidden_dim = int(self.config.method.get("domain_hidden_dim", 256))
        bottleneck_dim = int(self.config.method.get("bottleneck_dim", 0))
        film_scale = float(self.config.method.get("film_scale", 4.0))
        score_hidden_dim = int(self.config.method.get("score_hidden_dim", 64))

        self.net = ODCFMNet(
            backbone_name=backbone_name,
            num_classes=self.num_classes,
            domain_hidden_dim=domain_hidden_dim,
            bottleneck_dim=bottleneck_dim,
            film_scale=film_scale,
            score_hidden_dim=score_hidden_dim,
        ).to(self.device)

        # Loss weights
        self.lambda_domain = float(self.config.method.get("lambda_domain", 1.0))
        self.lambda_tgt = float(self.config.method.get("lambda_tgt", 1.0))
        self.lambda_supcon = float(self.config.method.get("lambda_supcon", 0.0))
        self.supcon_tau = float(self.config.method.get("supcon_tau", 0.1))

        # Pseudo-labeling / target loss knobs
        self.pl_conf = float(self.config.method.get("pl_conf", 0.95))
        self.pl_gamma = float(self.config.method.get("pl_gamma", 1.0))
        self.tgt_margin = float(self.config.method.get("tgt_margin", 0.05))
        self.tgt_margin_known = float(self.config.method.get("tgt_margin_known", 0.0))
        self.gate_temp = float(self.config.method.get("gate_temp", 0.02))
        self.lambda_ab_reg = float(self.config.method.get("lambda_ab_reg", 0.1))
        self.unk_ramp_epochs = int(self.config.method.get("unk_ramp_epochs", 5))
        # Keep small by default; too large causes "always unknown" collapse early.
        self.unk_logit_alpha = float(self.config.method.get("unk_logit_alpha", 0.5))
        self.unk_proto_temp = float(self.config.method.get("unk_proto_temp", 0.1))
        self.w_unk_conf_power = float(self.config.method.get("w_unk_conf_power", 1.0))

        # Prototypes
        self.proto_momentum = float(self.config.method.get("proto_momentum", 0.9))
        self.src_prototypes = torch.zeros(self.num_classes - 1, self.net.feat_dim, device=self.device)
        self.src_proto_inited = torch.zeros(self.num_classes - 1, dtype=torch.bool, device=self.device)
        # One unknown prototype per domain-mode (mixture via z_domain soft assignment)
        m = max(int(self.config.method.get("num_modes", 2)), 1)
        self.unk_prototypes = nn.Parameter(_normalize(torch.randn(m, self.net.feat_dim, device=self.device), dim=1))

        # GMM / threshold smoothing
        self.gmm_min_samples = int(self.config.method.get("gmm_min_samples", 128))
        self.thr_clip_lo_q = float(self.config.method.get("thr_clip_lo_q", 0.3))
        self.thr_clip_hi_q = float(self.config.method.get("thr_clip_hi_q", 0.9))
        self.thr_ema_decay = float(self.config.method.get("thr_ema_decay", 0.9))
        self._thr_ema: Optional[float] = None
        self._thr: Optional[float] = None
        self.num_modes = int(self.config.method.get("num_modes", 2))
        self.mode_assign_temp = float(self.config.method.get("mode_assign_temp", 1.0))
        self._mode_centers: Optional[torch.Tensor] = None  # [M, D_domain]
        self._thr_modes: Optional[torch.Tensor] = None  # [M]

        # Cached GMM params (for fast posterior per batch)
        self._gmm_means: Optional[torch.Tensor] = None  # [2]
        self._gmm_vars: Optional[torch.Tensor] = None  # [2]
        self._gmm_weights: Optional[torch.Tensor] = None  # [2]
        self._unk_comp: Optional[int] = None

        # Criteria
        label_smoothing = float(self.config.method.get("label_smoothing", 0.1))
        self.criterion_task = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.criterion_domain = nn.CrossEntropyLoss()

        # Warmup behavior
        self._disable_rejection = False

    def _extract_z_and_domain(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (logits, z_feat, z_domain) for input x.
        z_feat is the FiLM-modulated feature used by the classifier.
        """
        h = self.net.extract_h(x)
        _, z_domain = self.net.domain(h)
        z = self.net.feat_bn(h)
        z = self.net.film(z, z_domain)
        logits = self.net.classifier(z)
        return logits, z, z_domain

    def _augment_unknown_logit(
        self,
        logits: torch.Tensor,
        z_feat: torch.Tensor,
        z_domain: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add a domain-conditioned similarity-based term to the unknown logit:
        unk_logit += alpha * cos(z_feat, unk_proto_mix(z_domain)).
        """
        if self.num_modes <= 1 or self._mode_centers is None or self.unk_prototypes is None:
            return logits
        g = self._mode_weights(z_domain)  # [B, M]
        u = _normalize(self.unk_prototypes, dim=1)  # [M, D]
        # mix prototype per sample: [B, D]
        u_mix = g @ u
        u_mix = _normalize(u_mix, dim=1)
        z = _normalize(z_feat, dim=1)
        sim = (z * u_mix).sum(dim=1)  # [B]
        unk_idx = self.num_classes - 1
        logits = logits.clone()
        logits[:, unk_idx] = logits[:, unk_idx] + float(self.unk_logit_alpha) * sim
        return logits

    @torch.no_grad()
    def _update_source_prototypes(self, x_src: torch.Tensor, y_src: torch.Tensor):
        h = self.net.extract_h(x_src)
        z = self.net.feat_bn(h)
        z = _normalize(z, dim=1)
        y = y_src.view(-1)
        num_known = self.num_classes - 1
        for c in range(num_known):
            m = (y == c)
            if not m.any():
                continue
            mean_c = z[m].mean(dim=0)
            if not bool(self.src_proto_inited[c].item()):
                self.src_prototypes[c].copy_(mean_c)
                self.src_proto_inited[c] = True
            else:
                self.src_prototypes[c].mul_(self.proto_momentum).add_(mean_c, alpha=1.0 - self.proto_momentum)

    def forward_for_eval(self, imgs: torch.Tensor) -> torch.Tensor:
        logits, _ = self.net(imgs)
        return logits

    @torch.no_grad()
    def _collect_target_scores(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._set_eval_mode()
        scores = []
        raw = []
        zds = []
        for imgs, _ in self.target_loader:
            imgs = imgs.to(self.device)
            h = self.net.extract_h(imgs)
            _, z_domain = self.net.domain(h)
            z = self.net.feat_bn(h)
            z = _normalize(z, dim=1)
            d = cosine_distance_to_prototypes(z, self.src_prototypes[: self.num_classes - 1])
            d_dc, _, _ = self.net.score(d, z_domain)
            raw.append(d.detach().cpu())
            scores.append(d_dc.detach().cpu())
            zds.append(z_domain.detach().cpu())
        return torch.cat(raw, dim=0), torch.cat(scores, dim=0), torch.cat(zds, dim=0)

    def _mode_weights(self, z_domain: torch.Tensor) -> torch.Tensor:
        """
        Soft assignment of a sample to domain-modes using distances to centers.
        Returns g: [B, M], rows sum to 1.
        """
        if self._mode_centers is None:
            # fallback to uniform
            B = z_domain.size(0)
            return torch.full((B, self.num_modes), 1.0 / float(self.num_modes), device=z_domain.device, dtype=z_domain.dtype)
        z = z_domain
        c = self._mode_centers.to(z.device, dtype=z.dtype)  # [M, D]
        # squared euclidean distances [B, M]
        d2 = (z.unsqueeze(1) - c.unsqueeze(0)).pow(2).sum(dim=2)
        temp = float(max(self.mode_assign_temp, 1e-6))
        return torch.softmax(-d2 / temp, dim=1)

    def _fit_gmm_and_threshold(self) -> float:
        """
        Domain-conditioned multi-threshold:
        1) cluster target z_domain into M modes (MiniBatchKMeans)
        2) fit a 2-GMM on d_dc per mode and compute thr_m
        3) also keep an overall thr (mean of thr_m) for logging/back-compat
        """
        d, d_dc, zds = self._collect_target_scores()
        if d_dc.numel() < self.gmm_min_samples or self.num_modes <= 1:
            thr = float(torch.quantile(d_dc, 0.8).item())
            self._thr = thr
            self._thr_modes = torch.tensor([thr], device=self.device)
            self._mode_centers = None
            return thr

        # 1) fit domain-mode clustering on z_domain (target-only)
        z_np = zds.numpy()
        km = MiniBatchKMeans(
            n_clusters=self.num_modes,
            random_state=0,
            batch_size=1024,
            n_init="auto",
            max_iter=200,
        )
        mode_ids = km.fit_predict(z_np)  # [N]
        centers = torch.from_numpy(km.cluster_centers_).to(self.device, dtype=torch.float32)
        self._mode_centers = centers

        # 2) fit per-mode 2-GMM on d_dc
        thr_modes = []
        mode_counts = []
        for m in range(self.num_modes):
            idx = np.where(mode_ids == m)[0]
            mode_counts.append(int(idx.size))
            if idx.size < max(16, self.gmm_min_samples // 4):
                # fallback to global quantile if too few samples for stable per-mode GMM
                thr_m = float(torch.quantile(d_dc, 0.8).item())
                thr_modes.append(thr_m)
                continue

            x_m = d_dc[idx].numpy().reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
            gmm.fit(x_m)
            means = gmm.means_.reshape(-1)
            vars_ = gmm.covariances_.reshape(-1)
            weights = gmm.weights_.reshape(-1)
            low = int(np.argmin(means))
            high = int(np.argmax(means))
            thr_raw = _gmm_intersection_1d(
                mean_a=float(means[low]),
                var_a=float(vars_[low]),
                weight_a=float(weights[low]),
                mean_b=float(means[high]),
                var_b=float(vars_[high]),
                weight_b=float(weights[high]),
            )

            loq = float(min(max(self.thr_clip_lo_q, 0.0), 1.0))
            hiq = float(min(max(self.thr_clip_hi_q, 0.0), 1.0))
            if hiq < loq:
                loq, hiq = hiq, loq
            qlo = float(torch.quantile(d_dc[idx], loq).item())
            qhi = float(torch.quantile(d_dc[idx], hiq).item())
            thr_clip = float(min(max(float(thr_raw), qlo), qhi))
            thr_modes.append(thr_clip)

        thr_modes_t = torch.tensor(thr_modes, device=self.device, dtype=torch.float32)

        # 3) EMA smoothing on the *average* threshold (stability) and shift thr_modes accordingly
        thr_avg = float(thr_modes_t.mean().item())
        if self._thr_ema is None:
            thr_ema = thr_avg
        else:
            a = float(min(max(self.thr_ema_decay, 0.0), 1.0))
            thr_ema = a * float(self._thr_ema) + (1.0 - a) * thr_avg
        self._thr_ema = thr_ema

        # Keep relative offsets between modes but anchor to EMA mean
        thr_modes_t = thr_modes_t - thr_modes_t.mean() + float(thr_ema)
        self._thr_modes = thr_modes_t
        self._thr = float(thr_ema)

        # diagnostics
        try:
            d_q05 = float(torch.quantile(d, 0.05).item())
            d_q50 = float(torch.quantile(d, 0.50).item())
            d_q95 = float(torch.quantile(d, 0.95).item())
            q05 = float(torch.quantile(d_dc, 0.05).item())
            q50 = float(torch.quantile(d_dc, 0.50).item())
            q95 = float(torch.quantile(d_dc, 0.95).item())
            logger.info(
                f"GMM(d_dc) fit | modes={self.num_modes} mode_counts={mode_counts} thr_modes={thr_modes_t.detach().cpu().tolist()} "
                f"| d[q05,q50,q95]=[{d_q05:.3f},{d_q50:.3f},{d_q95:.3f}] "
                f"| d_dc[q05,q50,q95]=[{q05:.3f},{q50:.3f},{q95:.3f}] "
                f"thr_ema={thr_ema:.3f} clip_q=[{self.thr_clip_lo_q:.2f},{self.thr_clip_hi_q:.2f}]"
            )
        except Exception:
            pass

        return float(self._thr)

    def _unknown_posterior(self, d_dc: torch.Tensor) -> torch.Tensor:
        if self._gmm_means is None or self._gmm_vars is None or self._gmm_weights is None or self._unk_comp is None:
            # fallback: do a quick fit on CPU for this batch (should rarely happen)
            scores_cpu = d_dc.detach().float().cpu().numpy().reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0).fit(scores_cpu)
            means = gmm.means_.reshape(-1)
            unk_comp = int(np.argmax(means))
            post = gmm.predict_proba(scores_cpu)[:, unk_comp]
            return torch.from_numpy(post).to(d_dc.device, dtype=d_dc.dtype)

        x = d_dc.float()
        means = self._gmm_means.to(x.device, dtype=x.dtype)
        vars_ = self._gmm_vars.to(x.device, dtype=x.dtype)
        w = self._gmm_weights.to(x.device, dtype=x.dtype)
        p0 = w[0] * _gaussian_pdf(x, means[0], vars_[0])
        p1 = w[1] * _gaussian_pdf(x, means[1], vars_[1])
        denom = (p0 + p1).clamp_min(1e-12)
        post1 = p1 / denom
        post0 = p0 / denom
        return post1 if int(self._unk_comp) == 1 else post0

    def _compute_hscore(self, preds: torch.Tensor, labels: torch.Tensor):
        unknown_label = self.unknown_label
        known_mask = labels != unknown_label
        unknown_mask = labels == unknown_label

        known_acc = (preds[known_mask] == labels[known_mask]).float().mean().item() if known_mask.any() else 0.0
        unk_acc = (preds[unknown_mask] == unknown_label).float().mean().item() if unknown_mask.any() else 0.0
        h = (2 * known_acc * unk_acc / (known_acc + unk_acc)) if (known_acc + unk_acc) > 0 else 0.0
        return 100 * known_acc, 100 * unk_acc, 100 * h

    def evaluate(self):
        self._set_eval_mode()
        # Explicit unknown (K+1): directly argmax over all logits.
        num_known = self.num_classes - 1
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in self.target_test_loader:
                imgs = imgs.to(self.device)
                h = self.net.extract_h(imgs)
                _, z_domain = self.net.domain(h)
                z = self.net.feat_bn(h)
                z = _normalize(z, dim=1)
                logits = self.forward_for_eval(imgs)
                logits = self._augment_unknown_logit(logits, z_feat=z, z_domain=z_domain)
                pred = logits.argmax(dim=1).detach().cpu()
                all_preds.append(pred)
                all_labels.append(labels)
        return self._compute_hscore(torch.cat(all_preds), torch.cat(all_labels))

    def train(self):
        warmup_epochs = int(self.config.method.get("warmup_epochs", 10))
        joint_epochs = int(self.config.method.get("epochs", 30))
        lr = float(self.config.method.get("lr", 0.005))

        optimizer = optim.SGD(
            [
                {"params": self.net.backbone.parameters(), "lr": lr * 0.1},
                {"params": self.net.feat_bn.parameters(), "lr": lr},
                {"params": self.net.domain.parameters(), "lr": lr},
                {"params": self.net.film.parameters(), "lr": lr},
                {"params": self.net.score.parameters(), "lr": lr},
                {"params": self.net.classifier.parameters(), "lr": lr},
                {"params": [self.unk_prototypes], "lr": lr},
            ],
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
        )

        total_iters = (warmup_epochs + joint_epochs) * len(self.source_loader)
        warmup_iters = warmup_epochs * len(self.source_loader)

        def lr_lambda(step: int) -> float:
            if step < warmup_iters:
                return float(step) / max(1, warmup_iters)
            progress = (step - warmup_iters) / max(1, total_iters - warmup_iters)
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        save_dir = Path("checkpoints")
        save_dir.mkdir(parents=True, exist_ok=True)
        best_path = save_dir / "best.pth"
        best_h = -1.0

        logger.info(f"ODCFM (rewrite) | warmup={warmup_epochs} joint={joint_epochs}")

        # ----------------- Warmup (source + domain only) -----------------
        for ep in range(warmup_epochs):
            self._disable_rejection = True
            self._thr = None
            self._gmm_means = self._gmm_vars = self._gmm_weights = None
            self._unk_comp = None
            self._thr_modes = None
            self._mode_centers = None
            self.net.train()

            meters = {k: AverageMeter() for k in ["task", "domain", "total"]}
            tgt_iter = cycle(self.target_loader)

            for x_src, y_src in self.source_loader:
                x_tgt, _ = next(tgt_iter)
                x_src, y_src = x_src.to(self.device), y_src.to(self.device)
                x_tgt = x_tgt.to(self.device)

                self._update_source_prototypes(x_src, y_src)

                optimizer.zero_grad()
                logits_src, dom_src = self.net(x_src)
                _, dom_tgt = self.net(x_tgt)

                loss_task = self.criterion_task(logits_src[:, : self.num_classes - 1], y_src)
                dom_logits = torch.cat([dom_src, dom_tgt], dim=0)
                dom_labels = torch.cat(
                    [
                        torch.zeros(x_src.size(0), dtype=torch.long, device=self.device),
                        torch.ones(x_tgt.size(0), dtype=torch.long, device=self.device),
                    ],
                    dim=0,
                )
                loss_domain = self.criterion_domain(dom_logits, dom_labels)
                if self.lambda_supcon > 0:
                    _, z_src, _ = self._extract_z_and_domain(x_src)
                    loss_supcon = supervised_contrastive_loss(z_src, y_src, tau=self.supcon_tau)
                else:
                    loss_supcon = torch.zeros((), device=self.device)
                loss = loss_task + self.lambda_domain * loss_domain + self.lambda_supcon * loss_supcon

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                optimizer.step()
                scheduler.step()

                meters["task"].update(loss_task.item())
                meters["domain"].update(loss_domain.item())
                meters["total"].update(loss.item())

            k, u, h = self.evaluate()
            if h > best_h:
                best_h = h
                self.save_checkpoint(best_path)
            logger.info(
                f"Warmup {ep+1:02d} | ts={meters['task'].avg:.3f} dm={meters['domain'].avg:.3f} "
                f"| K={k:.1f}% U={u:.1f}% H={h:.1f}% (best={best_h:.1f}%)"
            )

        # ----------------- Joint (enable target loss + GMM) -----------------
        for ep in range(joint_epochs):
            self._disable_rejection = False
            thr = self._fit_gmm_and_threshold()
            self.net.train()

            meters = {k: AverageMeter() for k in ["task", "domain", "tgt", "total"]}
            tgt_iter = cycle(self.target_loader)
            num_known = self.num_classes - 1

            for x_src, y_src in self.source_loader:
                x_tgt, _ = next(tgt_iter)
                x_src, y_src = x_src.to(self.device), y_src.to(self.device)
                x_tgt = x_tgt.to(self.device)

                self._update_source_prototypes(x_src, y_src)

                optimizer.zero_grad()

                logits_src, z_src, z_domain_src = self._extract_z_and_domain(x_src)
                logits_tgt, z_tgt, z_domain_t = self._extract_z_and_domain(x_tgt)
                # domain logits still from detached pathway for stability
                dom_src, _ = self.net.domain(self.net.extract_h(x_src).detach())
                dom_tgt, _ = self.net.domain(self.net.extract_h(x_tgt).detach())

                # L_src_task (only known classes)
                loss_task = self.criterion_task(logits_src[:, :num_known], y_src)

                # L_domain
                dom_logits = torch.cat([dom_src, dom_tgt], dim=0)
                dom_labels = torch.cat(
                    [
                        torch.zeros(x_src.size(0), dtype=torch.long, device=self.device),
                        torch.ones(x_tgt.size(0), dtype=torch.long, device=self.device),
                    ],
                    dim=0,
                )
                loss_domain = self.criterion_domain(dom_logits, dom_labels)

                # Optional 4th term: source SupCon to stabilize class separation
                if self.lambda_supcon > 0:
                    loss_supcon = supervised_contrastive_loss(z_src, y_src, tau=self.supcon_tau)
                else:
                    loss_supcon = torch.zeros((), device=self.device)

                # L_tgt_unified (explicit unknown K+1)
                # Build domain-conditioned score and sample-wise threshold thr_x (detached).
                z_t = _normalize(z_tgt, dim=1)
                d = cosine_distance_to_prototypes(z_t, self.src_prototypes[:num_known])
                d_dc, a_s, b_s = self.net.score(d, z_domain_t)
                temp = float(max(self.gate_temp, 1e-4))
                if self._thr_modes is not None and self._thr_modes.numel() == self.num_modes and self._mode_centers is not None:
                    g = self._mode_weights(z_domain_t)
                    thr_x = (g * self._thr_modes.to(g.device, dtype=g.dtype).unsqueeze(0)).sum(dim=1)
                else:
                    thr_x = torch.full_like(d_dc.detach(), float(thr))
                # unknownness from distance threshold, but damped by confidence in known classes
                logits_tgt_aug = self._augment_unknown_logit(logits_tgt, z_feat=z_t, z_domain=z_domain_t)
                probs_known = torch.softmax(logits_tgt_aug[:, :num_known], dim=1)
                conf_known, pseudo = probs_known.max(dim=1)
                w_unk_dist = torch.sigmoid((d_dc.detach() - thr_x.detach()) / temp)
                w_unk_conf = (1.0 - conf_known.detach()).clamp(0.0, 1.0).pow(float(self.w_unk_conf_power))
                w_unk = (w_unk_dist * w_unk_conf).clamp(0.0, 1.0).to(d_dc.device, dtype=d_dc.dtype)
                w_known = 1.0 - w_unk

                # ramp unknown-related terms to avoid early collapse
                if self.unk_ramp_epochs > 0:
                    unk_ramp = min(1.0, float(ep + 1) / float(self.unk_ramp_epochs))
                else:
                    unk_ramp = 1.0

                # known self-training (soft-weighted) -- IMPORTANT: use full (K+1) logits
                # so the gradient explicitly pushes down the unknown logit for known pseudo-labels.
                denom = max(float(self.pl_conf), 1e-6)
                weight_conf = (conf_known / denom).clamp(0.0, 1.0).pow(float(self.pl_gamma))
                per_ce_known = F.cross_entropy(logits_tgt_aug, pseudo, reduction="none")
                loss_known = (w_known * weight_conf * per_ce_known).mean()

                # unknown training: push to explicit unknown class (soft-weighted)
                unk_idx = self.num_classes - 1
                unk_target = torch.full((x_tgt.size(0),), unk_idx, dtype=torch.long, device=self.device)
                per_ce_unk = F.cross_entropy(logits_tgt_aug, unk_target, reduction="none")
                loss_unk = (w_unk * per_ce_unk).mean()

                # unknown prototype alignment (domain-conditioned via mixture) for samples with high w_unk
                if self._mode_centers is not None and self.num_modes > 1:
                    g = self._mode_weights(z_domain_t)  # [B,M]
                    u = _normalize(self.unk_prototypes, dim=1)
                    z_norm = _normalize(z_t, dim=1)
                    sim_um = z_norm @ u.t()  # [B,M]
                    a_um = torch.softmax(sim_um / float(self.unk_proto_temp), dim=1)
                    dist_u = 1.0 - (a_um * sim_um).sum(dim=1)
                    loss_align = (w_unk * dist_u).mean()
                else:
                    loss_align = torch.zeros((), device=self.device)

                loss_tgt = loss_known + unk_ramp * (loss_unk + loss_align)

                # keep calibration near identity to avoid pathological scaling
                ab_reg = ((a_s - 1.0).pow(2) + b_s.pow(2)).mean()
                loss_tgt = loss_tgt + float(self.lambda_ab_reg) * ab_reg

                loss = (
                    loss_task
                    + self.lambda_domain * loss_domain
                    + self.lambda_tgt * loss_tgt
                    + self.lambda_supcon * loss_supcon
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                optimizer.step()
                scheduler.step()

                meters["task"].update(loss_task.item())
                meters["domain"].update(loss_domain.item())
                meters["tgt"].update(loss_tgt.item())
                meters["total"].update(loss.item())

            # pseudo-label diagnostics (epoch-level; last batch is enough for trend signal)
            try:
                pl_rate = float((weight_conf > 0).float().mean().item())
                conf_q50 = float(torch.quantile(conf_known.detach(), 0.5).item())
                conf_q90 = float(torch.quantile(conf_known.detach(), 0.9).item())
                wconf_q50 = float(torch.quantile(weight_conf.detach(), 0.5).item())
                wconf_q90 = float(torch.quantile(weight_conf.detach(), 0.9).item())
                logger.info(
                    f"PL | active_rate={pl_rate:.3f} conf[q50,q90]=[{conf_q50:.3f},{conf_q90:.3f}] "
                    f"wconf[q50,q90]=[{wconf_q50:.3f},{wconf_q90:.3f}] pl_conf={self.pl_conf:.2f} gamma={self.pl_gamma:.2f}"
                )
            except Exception:
                pass

            k, u, h = self.evaluate()
            if h > best_h:
                best_h = h
                self.save_checkpoint(best_path)

            # diagnostics: a/b stats from last batch (safe best-effort)
            try:
                a_mean = float(a_s.mean().item())
                a_std = float(a_s.std().item())
                b_mean = float(b_s.mean().item())
                b_std = float(b_s.std().item())
                w_mean = float(w_unk.mean().item())
                w_q50 = float(torch.quantile(w_unk, 0.5).item())
                w_q90 = float(torch.quantile(w_unk, 0.9).item())
                logger.info(
                    f"Diag | a={a_mean:.3f}±{a_std:.3f} b={b_mean:.3f}±{b_std:.3f} "
                    f"| w_unk_gate[mean,q50,q90]=[{w_mean:.3f},{w_q50:.3f},{w_q90:.3f}]"
                )
            except Exception:
                pass

            logger.info(
                f"Joint {ep+1:02d} | ts={meters['task'].avg:.3f} dm={meters['domain'].avg:.3f} "
                f"tgt={meters['tgt'].avg:.3f} | thr_avg={thr:.3f} | "
                f"K={k:.1f}% U={u:.1f}% H={h:.1f}% (best={best_h:.1f}%)"
            )

        if best_path.exists():
            self.load_checkpoint(best_path)
            logger.info(f"Loaded best model (H={best_h:.2f}%) from {best_path}")

    def save_checkpoint(self, path):
        torch.save(
            {
                "method": "odcfm",
                "model": self.net.state_dict(),
                "thr": self._thr,
                "thr_ema": self._thr_ema,
                "thr_modes": None if self._thr_modes is None else self._thr_modes.detach().cpu(),
                "mode_centers": None if self._mode_centers is None else self._mode_centers.detach().cpu(),
                "gmm_means": None if self._gmm_means is None else self._gmm_means.detach().cpu(),
                "gmm_vars": None if self._gmm_vars is None else self._gmm_vars.detach().cpu(),
                "gmm_weights": None if self._gmm_weights is None else self._gmm_weights.detach().cpu(),
                "unk_comp": self._unk_comp,
                "src_prototypes": self.src_prototypes.detach().cpu(),
                "src_proto_inited": self.src_proto_inited.detach().cpu(),
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        if "model" in ckpt:
            self.net.load_state_dict(ckpt["model"])
        else:
            self.net.load_state_dict(ckpt)

        self._thr = ckpt.get("thr")
        self._thr_ema = ckpt.get("thr_ema")
        if ckpt.get("thr_modes") is not None:
            self._thr_modes = ckpt["thr_modes"].to(self.device)
        if ckpt.get("mode_centers") is not None:
            self._mode_centers = ckpt["mode_centers"].to(self.device)
        self._unk_comp = ckpt.get("unk_comp")
        if ckpt.get("gmm_means") is not None:
            self._gmm_means = ckpt["gmm_means"].to(self.device)
        if ckpt.get("gmm_vars") is not None:
            self._gmm_vars = ckpt["gmm_vars"].to(self.device)
        if ckpt.get("gmm_weights") is not None:
            self._gmm_weights = ckpt["gmm_weights"].to(self.device)
        if ckpt.get("src_prototypes") is not None:
            self.src_prototypes.copy_(ckpt["src_prototypes"].to(self.device))
        if ckpt.get("src_proto_inited") is not None:
            self.src_proto_inited.copy_(ckpt["src_proto_inited"].to(self.device))
        logger.info(f"Model loaded from {path}")
