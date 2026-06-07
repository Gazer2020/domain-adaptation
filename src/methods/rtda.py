import logging
import numpy as np
import faiss
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

from methods.registry import register_solver
from methods.base_solver import BaseSolver
from models.backbones import get_backbone
from utils import AverageMeter, configure_faiss_runtime, cycle

logger = logging.getLogger(__name__)


# ---------------- Utilities and Losses ----------------

class Accumulator(dict):
    def __init__(self, name_or_names, accumulate_fn=np.concatenate):
        super(Accumulator, self).__init__()
        self.names = [name_or_names] if isinstance(name_or_names, str) else name_or_names
        self.accumulate_fn = accumulate_fn
        for name in self.names:
            self.__setitem__(name, [])

    def updateData(self, scope):
        for name in self.names:
            if hasattr(scope[name], 'shape') and len(scope[name].shape) > 0 and scope[name].shape[-1] > 0:
                self.__getitem__(name).append(scope[name])
            elif isinstance(scope[name], list) and len(scope[name]) > 0:
                self.__getitem__(name).append(scope[name])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_tb:
            return False
        for name in self.names:
            if len(self.__getitem__(name)) > 0:
                self.__setitem__(name, self.accumulate_fn(self.__getitem__(name)))
        return True


def variable_to_numpy(x):
    return x.detach().cpu().numpy()


def to_np(x):
    return x.squeeze().cpu().detach().numpy()


def inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    return initial_lr * ((1 + gamma * min(1.0, step / float(max_iter))) ** (-power))


def aToBSheduler(step, A, B, gamma=10, max_iter=10000):
    ans = A + (2.0 / (1 + np.exp(-gamma * step * 1.0 / max_iter)) - 1.0) * (B - A)
    return float(ans)


def CrossEntropyLoss(label, predict_prob, epsilon=1e-12):
    if label.shape != predict_prob.shape:
        label = torch.zeros_like(predict_prob).scatter(1, label.unsqueeze(1), 1)
    N, C = label.size()
    ce = -label * torch.log(predict_prob + epsilon)
    return torch.sum(ce) / float(N)


def BCELossForMultiClassification(label, predict_prob, instance_level_weight=None, epsilon=1e-12):
    N, C = label.size()
    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
            
    bce = -label * torch.log(predict_prob + epsilon) - (1.0 - label) * torch.log(1.0 - predict_prob + epsilon)
    return torch.sum(instance_level_weight * bce) / float(N)


def EntropyLoss(predict_prob, instance_level_weight=None, epsilon=1e-20):
    N, C = predict_prob.size()
    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)

    entropy = -predict_prob * torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * entropy) / float(N)


class OptimWithSheduler:
    def __init__(self, optimizer, scheduler_func):
        self.optimizer = optimizer
        self.scheduler_func = scheduler_func
        self.global_step = 0.0
        for g in self.optimizer.param_groups:
            g['initial_lr'] = g['lr']

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        for g in self.optimizer.param_groups:
            g['lr'] = self.scheduler_func(step=self.global_step, initial_lr=g['initial_lr'])
        self.optimizer.step()
        self.global_step += 1

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state["optimizer"])
        self.global_step = float(state.get("global_step", 0.0))


class OptimizerManager:
    def __init__(self, optims):
        self.optims = optims
    def __enter__(self):
        for op in self.optims:
            op.zero_grad()
    def __exit__(self, exceptionType, exception, exceptionTraceback):
        if exceptionType is None:
            for op in self.optims:
                op.step()
        self.optims = None
        return False

# ---------------- Network Modules ----------------

class Centroids(object):
    def __init__(self, class_num, dim, device):
        self.class_num = class_num
        self.src_ctrs = torch.ones((class_num, dim)).to(device) * 1e-10
        self.tgt_ctrs = torch.ones((class_num, dim + 1)).to(device) * 1e-10
        self.dim = dim
        self.device = device

    def get_centroids(self):
        return self.src_ctrs, self.tgt_ctrs

    @torch.no_grad()
    def update(self, pred_s, pred_t, label_s, label_unk=None):
        self.upd_src_centroids(pred_s, label_s)
        self.upd_tgt_centroids(pred_t, label_unk)

    @torch.no_grad()
    def upd_src_centroids(self, probs, labels):
        for i in range(self.class_num):
            data_idx = np.argwhere(labels[:, i] == 1)[:, 0]
            if len(data_idx) > 0:
                new_centroid = torch.mean(torch.tensor(probs[data_idx, :self.dim]), 0).squeeze()
                self.src_ctrs[i, :] = new_centroid.to(self.device)

    @torch.no_grad()
    def upd_tgt_centroids(self, probs, labels):
        if labels is None:
            return
        for i in range(self.class_num):
            data_idx = np.argwhere(labels == i)
            if len(data_idx) > 0:
                new_centroid = torch.mean(torch.tensor(probs[data_idx]), 0).squeeze()
                self.tgt_ctrs[i, :] = new_centroid.to(self.device)


class CLS(nn.Module):
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256, temp=0.05):
        super(CLS, self).__init__()
        self.temp = 1
        if bottle_neck_dim:
            self.bottleneck = nn.Linear(in_dim, bottle_neck_dim)
            self.weight1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.fc = nn.Linear(bottle_neck_dim, out_dim, bias=False)
            
            self.main = nn.Sequential(
                self.bottleneck,
                nn.Sequential(
                    nn.BatchNorm1d(bottle_neck_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                    self.fc
                ),
                nn.Softmax(dim=-1)
            )
        else:
            self.fc = nn.Linear(in_dim, out_dim)
            self.main = nn.Sequential(
                self.fc,
                nn.Softmax(dim=-1)
            )

    def forward(self, x):
        out = [x]
        for i, module in enumerate(self.main.children()):
            if i == 0:
                x = module(x)
                x = x / torch.norm(x, dim=-1, keepdim=True)
            else:
                x = module(x)
            out.append(x)
        out[-2] = out[-2] / self.temp
        out[-1] = nn.Softmax(dim=-1)(out[-2])
        return out
    
    def virt_forward(self, K, feature_source, logits, target):
        if self.training:
            with torch.no_grad():
                W_yi = torch.gather(self.fc.weight, 0, target.unsqueeze(1).expand(target.size(0), self.fc.weight.size(1)))   
                W_virt = torch.norm(W_yi, dim=1).unsqueeze(-1).unsqueeze(-1) * ((K / torch.norm(K, dim=1).unsqueeze(-1)).unsqueeze(0))
            vir = torch.bmm(W_virt, feature_source.unsqueeze(-1)).squeeze(-1)
            logits = torch.cat([logits, vir], dim=-1)
            x = nn.Softmax(-1)(logits)
        return x


class GradientReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coeff, input):
        ctx.coeff = coeff
        return input

    @staticmethod
    def backward(ctx, grad_outputs):
        coeff = ctx.coeff
        return None, -coeff * grad_outputs


class GradientReverseModule(nn.Module):
    def __init__(self, scheduler):
        super(GradientReverseModule, self).__init__()
        self.scheduler = scheduler
        self.global_step = 0.0
        self.coeff = 0.0
        self.grl = GradientReverseLayer.apply

    def forward(self, x):
        self.coeff = self.scheduler(self.global_step)
        self.global_step += 1.0
        return self.grl(self.coeff, x)


class AdversarialNetwork(nn.Module):
    def __init__(self):
        super(AdversarialNetwork, self).__init__()
        self.main = nn.Sequential()
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x):
        x = self.grl(x)
        for module in self.main.children():
            x = module(x)
        return x


class LargeAdversarialNetwork(AdversarialNetwork):
    def __init__(self, in_feature):
        super(LargeAdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, 1024)
        self.ad_layer2 = nn.Linear(1024, 1024)
        self.ad_layer3 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

        self.main = nn.Sequential(
            self.ad_layer1,
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            self.ad_layer2,
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            self.ad_layer3,
            self.sigmoid
        )


# ---------------- RTDA Solver ----------------

@register_solver("rtda")
class RTDASolver(BaseSolver):
    def build_model(self):
        faiss_threads = configure_faiss_runtime(self.config)
        self.shared_classes = self.class_info["num_classes"]
        self.all_classes = self.shared_classes + 2  # 1 unknown + 1 virtual cluster handling
        
        backbone_name = self.config.method.get("backbone", "resnet50")
        self.feature_extractor = get_backbone(backbone_name)
        if hasattr(self.feature_extractor, "fc"):
            self.in_features = self.feature_extractor.fc.in_features
            self.feature_extractor.fc = nn.Identity()
        else:
            self.in_features = self.feature_extractor.classifier[6].in_features
            self.feature_extractor.classifier[6] = nn.Identity()
            
        self.cls = CLS(self.in_features, self.all_classes, bottle_neck_dim=256)
        # self.net logic to support forward_for_eval easily
        self.net = nn.Sequential(self.feature_extractor, self.cls).to(self.device)
        self.discriminator = LargeAdversarialNetwork(256).to(self.device)
        logger.info("RTDA FAISS runtime | threads=%d", faiss_threads)

    def forward_for_eval(self, imgs):
        # Override BaseSolver's forward_for_eval
        # CLS returns a list: [..., unnormalized_logits, softmax_probs]
        # Our base solver expects unnormalized logits!
        outputs = self.net(imgs)
        logits = outputs[-2] 
        return logits

    def _set_train_mode(self):
        self.net.train()
        self.discriminator.train()

    def _set_eval_mode(self):
        self.net.eval()
        self.discriminator.eval()

    def _fast_initial_classifier_weight(self, source_loader, target_loader):
        logger.info("Initializing fast classification weights using KMeans clustering...")
        self.net.eval()
        
        # We need to compute features
        with torch.no_grad():
            with Accumulator(['fs', 'ft', 'ls']) as ProbRecorder:
                for src, tgt in zip(source_loader, target_loader):
                    im_source, label_source = self._to_device(src[0]), self._to_device(src[1])
                    im_target, _ = self._to_device(tgt[0]), self._to_device(tgt[1])

                    _, feature_source, _, _ = self.net(im_source)
                    _, feature_target, _, _ = self.net(im_target)
                    
                    fs = variable_to_numpy(feature_source)
                    ft = variable_to_numpy(feature_target)
                    ls = variable_to_numpy(label_source)
                    ProbRecorder.updateData({'fs':fs, 'ft':ft, 'ls':ls})
        
        fs_numpy = ProbRecorder['fs']
        ls_numpy = ProbRecorder['ls']
        ft_numpy = ProbRecorder['ft']

        s_centroids = []
        for i in range(self.shared_classes):
            mask = (ls_numpy == i)
            if np.sum(mask) > 0:
                s_centroids.append(fs_numpy[mask].mean(axis=0))
            else:
                # Fallback if a class isn't in this iteration (rare)
                s_centroids.append(np.zeros(fs_numpy.shape[-1]))
        s_centroids = np.stack(s_centroids, axis=0)

        K_cluster = self.config.method.K_cluster
        faiss_kmeans = faiss.Kmeans(256, int(K_cluster), niter=800, verbose=False, min_points_per_centroid=1, gpu=False)
        faiss_kmeans.train(ft_numpy)
        t_centroids = faiss_kmeans.centroids
        
        cost = np.linalg.norm(s_centroids[:, None, :] - t_centroids[None, :, :], axis=-1)
        _, t_match = linear_sum_assignment(cost)
        nomatch = []
        for i in range(K_cluster):
            if i not in t_match:
                nomatch.append(t_centroids[i])
        nomatch = np.stack(nomatch, axis=0)

        fcweight = np.concatenate([s_centroids, nomatch], axis=0)
        
        for key, v in self.net.state_dict().items():
            if '1.fc.weight' in key:
                cost_w = np.linalg.norm(fcweight[:, None, :] - v.cpu().numpy()[None, :, :], axis=-1)
                _, t_match_w = linear_sum_assignment(cost_w)
                param = torch.from_numpy(v.cpu().numpy()[t_match_w]).to(self.device).detach().clone()
                self.net.state_dict()[key].copy_(param)

        # Store nomatch for virt_forward
        self.nomatch = torch.from_numpy(nomatch).to(self.device).detach().clone()

    def predict_with_rejection(self, preds: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        # Override to use standard OSDA prediction mapped correctly
        # The class index self.shared_classes corresponds to the unknown bucket
        
        # `preds` contains the argmax prediction
        rejected_mask = preds >= self.shared_classes
        final_preds = preds.clone()
        final_preds[rejected_mask] = self.class_info["unknown_label"]
        
        return final_preds

    def extra_training_state_dict(self):
        state = super().extra_training_state_dict()
        if hasattr(self, "nomatch"):
            state["nomatch"] = self.nomatch
        if hasattr(self, "all_centroids"):
            state["src_centroids"] = self.all_centroids.src_ctrs
            state["tgt_centroids"] = self.all_centroids.tgt_ctrs
        return state

    def load_extra_training_state_dict(self, state):
        super().load_extra_training_state_dict(state)
        self._resume_nomatch = state.get("nomatch")
        self._resume_src_centroids = state.get("src_centroids")
        self._resume_tgt_centroids = state.get("tgt_centroids")

    def train(self):
        max_epochs = self.config.method.epochs
        warmiter = self.config.method.warm_up_epoch
        K_cluster = self.config.method.K_cluster
        learning_rate = self.config.method.lr
        momentum = float(self.config.method.get("momentum", 0.9))
        weight_decay = float(self.config.method.get("weight_decay", 5e-4))
        nesterov = self._is_truthy(self.config.method.get("nesterov", True))
        head_lr_mult = float(self.config.method.get("head_lr_mult", 10.0))
        
        # Limit len to smaller of the two loops or large bounds
        max_len = max(len(self.source_loader), len(self.target_loader))
        max_iter = max_epochs * max_len
        
        self.all_centroids = Centroids(class_num=self.shared_classes, dim=self.shared_classes, device=self.device)
        if self._resume_epoch > 0 and getattr(self, "_resume_nomatch", None) is not None:
            self.nomatch = self._resume_nomatch.to(self.device)
            if self._resume_src_centroids is not None:
                self.all_centroids.src_ctrs.copy_(self._resume_src_centroids.to(self.device))
            if self._resume_tgt_centroids is not None:
                self.all_centroids.tgt_ctrs.copy_(self._resume_tgt_centroids.to(self.device))
        else:
            self._fast_initial_classifier_weight(self.source_loader, self.target_loader)
        
        scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=max_iter)
        
        optimizer_kwargs = {
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
        }
        optimizer_feature_extractor = OptimWithSheduler(
            optim.SGD(self.feature_extractor.parameters(), lr=learning_rate, **optimizer_kwargs),
            scheduler,
        )
        optimizer_cls = OptimWithSheduler(
            optim.SGD(self.cls.parameters(), lr=learning_rate * head_lr_mult, **optimizer_kwargs),
            scheduler,
        )
        optimizer_discriminator = OptimWithSheduler(
            optim.SGD(
                self.discriminator.parameters(),
                lr=learning_rate * head_lr_mult,
                **optimizer_kwargs,
            ),
            scheduler,
        )
        self.register_training_state(
            classifier_optimizer=optimizer_cls,
            feature_optimizer=optimizer_feature_extractor,
            discriminator_optimizer=optimizer_discriminator,
        )
        
        best_hos = self._best_metric
        gmm = None

        for epoch in self._epoch_range(max_epochs):
            self._set_train_mode()
            
            src_iter = cycle(self.source_loader)
            tgt_iter = cycle(self.target_loader)
            
            loss_meter = AverageMeter()
            
            with Accumulator(['pred_s', 'pred_t', 'label_s', 'kl', 'fss', 'ftt']) as ProbRecorder:
                for i in range(max_len):
                    im_source, label_source = next(src_iter)
                    im_target, _ = next(tgt_iter)

                    im_source = self._to_device(im_source)
                    label_source = self._to_device(label_source)
                    # Create one-hot label since original RTDA logic expects it
                    label_s_one_hot = torch.zeros(label_source.shape[0], self.all_classes, device=self.device)
                    label_s_one_hot.scatter_(1, label_source.unsqueeze(1), 1)

                    im_target = self._to_device(im_target)
                    
                    _, feature_source, fc_source, predict_prob_source = self.net(im_source)
                    ft1, feature_target, fc_target, predict_prob_target = self.net(im_target)
                    
                    domain_prob_discriminator_1_source = self.discriminator(feature_source)
                    domain_prob_discriminator_1_target = self.discriminator(feature_target)
                    
                    s_ctds, t_ctds = self.all_centroids.get_centroids()  
                    _, pseudo_t_label = predict_prob_target[:, :self.shared_classes].max(1)
                    
                    kltarget = torch.nn.functional.kl_div((nn.Softmax(-1)(fc_target[:, :self.shared_classes])).log(), s_ctds[pseudo_t_label], reduction='none').sum(1).detach()
                    kltarget = torch.where(torch.isinf(kltarget), torch.full_like(kltarget, 10), kltarget)

                    if epoch <= 1 or gmm is None:
                        gmm = GaussianMixture(n_components=3, covariance_type='full').fit(to_np(kltarget)[:, None])
                    
                    known_cluster = np.argmin(gmm.means_)
                    unknown_cluster = np.argmax(gmm.means_)
                    gmm_index = gmm.predict(to_np(kltarget)[:, None])
                    
                    pred_s, pred_t, label_s, kl, fss, ftt = [
                        variable_to_numpy(x) for x in (nn.Softmax(-1)(fc_source[:, :self.shared_classes]),
                        predict_prob_target, label_s_one_hot, kltarget, feature_source, feature_target)
                    ]
                    ProbRecorder.updateData({'pred_s':pred_s, 'pred_t':pred_t, 'label_s':label_s, 'kl':kl, 'fss':fss, 'ftt':ftt})

                    weight_np = gmm.predict_proba(to_np(kltarget)[:, None])[:, known_cluster]
                    weight = torch.as_tensor(weight_np, device=self.device, dtype=kltarget.dtype).detach()
                    gmm_index_t = torch.as_tensor(gmm_index, device=self.device)
                    
                    if epoch <= 10:
                        weight = (weight > 0.8).to(weight.dtype).detach()
                        r = torch.nonzero(gmm_index_t != known_cluster).unsqueeze(-1)
                        topk = 16
                        if r.size()[0] > topk:
                            # Re-sort to pick topk indices
                            _, indices = torch.sort(kltarget.detach(), dim=0)
                            r = indices[-1 * topk:].unsqueeze(-1)
                    else:             
                        weight = (gmm_index_t == known_cluster).to(weight.dtype).detach()
                        r = torch.nonzero(gmm_index_t == unknown_cluster).unsqueeze(-1)
           
                    feature_otherep = torch.index_select(ft1, 0, r.view(-1))
                    
                    if r.size()[0] > 1:
                        _, feature_otherep, logits_otherep, predict_prob_otherep = self.cls(feature_otherep)
                        _, pseudo_index = predict_prob_otherep[:, self.shared_classes:].max(1)
                        pseudo_index = pseudo_index + self.shared_classes
                        pseudo_label = torch.zeros(r.size()[0], self.all_classes, device=self.device).scatter_(1, pseudo_index.unsqueeze(1), 1)
                        ce_ep = CrossEntropyLoss(pseudo_label, predict_prob_otherep)            
                    else:
                        ce_ep = torch.zeros((), device=self.device)
                       
                    ce = CrossEntropyLoss(label_s_one_hot, nn.Softmax(-1)(fc_source))

                    virtual_predict_prob_source = self.cls.virt_forward(self.nomatch, feature_source, fc_source, label_source)
                    p = torch.zeros([label_s_one_hot.shape[0], self.nomatch.size(0)], device=self.device)
                    v_label_source = torch.cat((label_s_one_hot, p), 1)
                    virtual_ce = CrossEntropyLoss(v_label_source, virtual_predict_prob_source)
            
                    entropy = EntropyLoss(predict_prob_target, instance_level_weight=weight.contiguous())

                    adv_loss = BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_source), predict_prob=domain_prob_discriminator_1_source)
                    adv_loss += BCELossForMultiClassification(label=torch.ones_like(domain_prob_discriminator_1_target), predict_prob=1 - domain_prob_discriminator_1_target, instance_level_weight=weight.contiguous())
                       
                    with OptimizerManager([optimizer_cls, optimizer_feature_extractor, optimizer_discriminator]):
                        if epoch <= warmiter:
                            loss = ce + virtual_ce
                        else:
                            loss = ce + 0.01 * virtual_ce + 0.3 * adv_loss + entropy + ce_ep 
                        loss.backward()
                        
                    loss_meter.update(loss.item())

            # Evaluate at the end of epoch
            acc = self.evaluate()
            if acc > best_hos:
                best_hos = acc
            self._training_global_step = int(optimizer_cls.global_step)
            self._maybe_save_best(acc, epoch + 1)
            self._log_epoch_summary(
                epoch + 1,
                max_epochs,
                metrics={"loss": loss_meter.avg},
                score=acc,
                best_score=best_hos,
                score_name="Score",
            )

            # Update centroids for the next epoch based on recorded features
            self.all_centroids.update(ProbRecorder['pred_s'], ProbRecorder['pred_t'], ProbRecorder['label_s'])

            s_centroids = []
            for i in range(self.shared_classes):
                s_centroids.append(ProbRecorder['fss'][np.nonzero(ProbRecorder['label_s'])[1] == i].mean(axis=0))
            s_centroids = np.stack(s_centroids, axis=0)

            faiss_kmeans = faiss.Kmeans(256, int(K_cluster), niter=800, verbose=False, min_points_per_centroid=1, gpu=False)
            faiss_kmeans.train(ProbRecorder['ftt'])      
            t_centroids = faiss_kmeans.centroids

            cost = np.linalg.norm(s_centroids[:, None, :] - t_centroids[None, :, :], axis=-1)
            _, t_match = linear_sum_assignment(cost)
            nomatch = []
            for i in range(K_cluster):
                if i not in t_match:
                    nomatch.append(t_centroids[i])
            nomatch = np.stack(nomatch, axis=0)
            self.nomatch = torch.from_numpy(nomatch).to(self.device).detach().clone()

            if epoch == warmiter:
                faiss_kmeans = faiss.Kmeans(256, int(self.all_classes), niter=800, verbose=False, min_points_per_centroid=1, gpu=False)
                faiss_kmeans.train(ProbRecorder['ftt'])

                t_centroids = faiss_kmeans.centroids
                cost = np.linalg.norm(s_centroids[:, None, :] - t_centroids[None, :, :], axis=-1)
                _, t_match = linear_sum_assignment(cost)
                
                init_unk_weight = []
                for i in range(self.all_classes):
                    if i not in t_match:
                        init_unk_weight.append(t_centroids[i])
                init_unk_weight = np.stack(init_unk_weight, axis=0)
                
                for key, v in self.net.state_dict().items():   
                    if '1.fc.weight' in key:
                        v.requires_grad = False
                        self.net.state_dict()[key].requires_grad = False
                        
                        vvnorm = (torch.norm(v, dim=-1)).mean().cpu().numpy()
                        init_unk_weight = init_unk_weight / np.linalg.norm(init_unk_weight, axis=-1, keepdims=True) * vvnorm
                        fcweight = np.concatenate([v[:self.shared_classes].clone().detach().cpu().numpy(), init_unk_weight], axis=0)
                        param = torch.from_numpy(fcweight).to(self.device).detach().clone()
                        self.net.state_dict()[key].copy_(param)  
                        
                        v.requires_grad = True
                        self.net.state_dict()[key].requires_grad = True
            
            if epoch <= 30:
                gmm = BayesianGaussianMixture(n_components=4, max_iter=800).fit(ProbRecorder['kl'][:, None])
            else:
                gmm = BayesianGaussianMixture(n_components=2, max_iter=800).fit(ProbRecorder['kl'][:, None])
            self._maybe_save_training_checkpoint(epoch + 1)
        if self._load_best_checkpoint_if_available():
            self._log_best_checkpoint_loaded("Score")
        self._log_training_complete(best_score=best_hos, score_name="Score")
        torch.cuda.empty_cache()
