import logging
import numpy as np
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

from methods.registry import register_solver
from methods.base_solver import BaseSolver
from models.backbones import get_backbone
from utils import AverageMeter, configure_faiss_runtime, cycle

logger = logging.getLogger(__name__)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class Embedding(nn.Module):
    def __init__(self, feature_dim, embed_dim=256, type=0):
        super(Embedding, self).__init__()
        self.bn1 = nn.BatchNorm1d(embed_dim, affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, embed_dim)
        self.bottleneck.apply(init_weights)
        self.bn_type = type
        self.fc1_mu = nn.Linear(embed_dim, embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim, affine=True)
        self.fc1_sig = nn.Linear(embed_dim, embed_dim)
        self.bn3 = nn.BatchNorm1d(embed_dim, affine=True)
        self.threshold = 0.9
        self.sig_act = F.softplus
        self.swish1 = nn.SiLU()
        self.swish2 = nn.SiLU()
        self.swish3 = nn.SiLU()
        self.swish4 = nn.SiLU()
        self.bn4 = nn.BatchNorm1d(embed_dim, affine=True)

    def forward(self, x):
        x = self.bottleneck(x)
        if self.bn_type == 1:
            x = self.bn1(x)
            x = self.swish1(x)

        x_mu = self.fc1_mu(x)
        if self.bn_type == 1:
            x_mu = self.swish2(self.bn2(x_mu))
        x_sig = self.fc1_sig(x)
        if self.bn_type == 1:
            x_sig = self.bn3(x_sig)
        x_sig = self.sig_act(x_sig) + 1e-8
        x = Normal(x_mu, x_sig)
        dist = torch.mean(self.relu2(self.threshold - x.entropy()))
        for i in range(1):
            x = x.rsample()
        x = self.swish4(self.bn4(x))
        return x, dist

class Classifier(nn.Module):
    def __init__(self, embed_dim, class_num, type="linear"):
        super(Classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = nn.utils.weight_norm(nn.Linear(embed_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(embed_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x, apply_softmax=True):
        x = self.fc(x)
        if apply_softmax:
            cls_out = torch.softmax(x, dim=1)
        else:
            cls_out = x
        return cls_out

class MLP(nn.Module):
    def __init__(self, n_inputs, n_outputs, mlp_width=128, mlp_depth=3, mlp_dropout=0.0, bn_type=0):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, mlp_width)
        self.dropout = nn.Dropout(mlp_dropout)
        self.hiddens = nn.ModuleList([
            nn.Linear(mlp_width, mlp_width)
            for _ in range(mlp_depth-2)])
        self.output = nn.Linear(mlp_width, n_outputs)
        self.n_outputs = n_outputs
        self.fc1_mu = nn.Linear(n_outputs, n_outputs)
        self.fc1_sig = nn.Linear(n_outputs, n_outputs)
        self.threshold = 0.9
        self.sig_act = F.softplus
        self.bn_type = bn_type
        self.swish1 = nn.SiLU()
        self.swish2 = nn.SiLU()
        self.swish3 = nn.SiLU()
        self.swish4 = nn.SiLU()
        self.bn1 = nn.BatchNorm1d(n_outputs, affine=True)
        self.bn2 = nn.BatchNorm1d(n_outputs, affine=True)
        self.bn3 = nn.BatchNorm1d(n_outputs, affine=True)
        self.bn4 = nn.BatchNorm1d(n_outputs, affine=True)

    def forward(self, x1):
        x = self.input(x1)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        intervention = self.output(x)
        int_rois = x1 + intervention
        if self.bn_type == 1:
            int_rois = self.bn1(int_rois)
            int_rois = self.swish1(int_rois)
        x_mu = self.fc1_mu(int_rois)
        if self.bn_type == 1:
            x_mu = self.swish2(self.bn2(x_mu))
        x_sig = self.fc1_sig(int_rois)
        if self.bn_type == 1:
            x_sig = self.bn3(x_sig)
        x_sig = self.sig_act(x_sig) + 1e-8
        x = Normal(x_mu, x_sig)
        dist = torch.mean(F.relu(self.threshold - x.entropy()))
        for i in range(1):
            x = x.rsample()
        x = self.swish4(self.bn4(x))
        return intervention, dist, x

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss

class COSDA(nn.Module):
    def __init__(self, backbone_arch, embed_feat_dim, class_num, mlp_width, mlp_depth, mlp_dropout, bn_type):
        super(COSDA, self).__init__()
        self.backbone_layer = get_backbone(backbone_arch)
        # Handle different backbone types
        if hasattr(self.backbone_layer, 'fc'):
            self.backbone_feat_dim = self.backbone_layer.fc.in_features
            self.backbone_layer.fc = nn.Identity()
        elif hasattr(self.backbone_layer, 'classifier'):
            self.backbone_feat_dim = self.backbone_layer.classifier[6].in_features
            self.backbone_layer.classifier[6] = nn.Identity()
        else:
            raise ValueError("Unsupported backbone")
            
        self.feat_embed_layer = Embedding(self.backbone_feat_dim, embed_feat_dim, type=bn_type)
        self.intervener = MLP(embed_feat_dim, embed_feat_dim, mlp_width, mlp_depth, mlp_dropout, bn_type=bn_type)
        self.class_layer = Classifier(embed_feat_dim, class_num, type='wn')

    def forward(self, input_imgs, apply_softmax=True):
        rois = self.backbone_layer(input_imgs)
        rois_c, v = self.feat_embed_layer(rois)
        y = self.class_layer(rois_c, apply_softmax)

        intervention, int_v, int_rois = self.intervener(rois_c)
        int_y = self.class_layer(int_rois, apply_softmax)
        return rois, v, rois_c, y, intervention, int_rois, int_v, int_y


def kl_normal(qm, qv, pm, pv):
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    kl = element_wise.mean()
    return kl

def kl_loss(m, v, y, prior_type='no conditional', known_class=0):
    if prior_type == 'conditional':
        pass # Not used by default in COSDA script for prior_type='no conditional'
    pm, pv = torch.zeros_like(m), torch.ones_like(m)
    return kl_normal(m, v * 0.0001, pm, pv * 0.0001)

def intervention_loss(intervention, int_epsilon):
    return torch.norm(torch.pow(intervention, 2) - int_epsilon)

def kl_anneal_function(epoch, times, step, total_annealing_step=10000):
    return min(1, 2*(epoch*times+step)/ total_annealing_step)

def ce_criterion(num_classes, config, hard_label, output, domain):
    if config.method.adaptation_type == "smooth":
        return CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=config.method.smooth, reduction=True)(output, hard_label)
    elif config.method.adaptation_type == "vanilla":
        return CrossEntropyLabelSmooth(num_classes=num_classes, epsilon=0.0, reduction=True)(output, hard_label)

def CalculateMean(features, labels, class_num):
    C = class_num
    A = features.size(1)
    avg_CxA = features.new_zeros(C, A)
    counts = features.new_zeros(C)
    avg_CxA.index_add_(0, labels, features)
    counts.index_add_(0, labels, torch.ones(labels.size(0), device=labels.device, dtype=features.dtype))
    avg_CxA = avg_CxA / counts.clamp_min(1.0).unsqueeze(1)
    return avg_CxA.detach()

def MO(mean_source_up1, features_target1, hard_label_bank, class_num):
    ce_crit = nn.CrossEntropyLoss()
    N = features_target1.size(0)
    C = class_num
    A = features_target1.size(1)

    norm_features_target1 = features_target1.norm(dim=1, keepdim=True)
    norm_features_target1 = torch.where(norm_features_target1 == 0, torch.ones_like(norm_features_target1), norm_features_target1)
    features_target = features_target1 / norm_features_target1

    norm_mean_source_up1 = mean_source_up1.norm(dim=1, keepdim=True)
    norm_mean_source_up1 = torch.where(norm_mean_source_up1 == 0, torch.ones_like(norm_mean_source_up1), norm_mean_source_up1)
    mean_source_up = mean_source_up1 / norm_mean_source_up1

    predict_gnn_target = hard_label_bank

    sourceMean_NxCxA = mean_source_up.expand(N, C, A)
    sourceMean_NxAxC = sourceMean_NxCxA.permute(0, 2, 1)
    features_target_Nx1xA = features_target.unsqueeze(1)
    g_mu = torch.bmm(features_target_Nx1xA, sourceMean_NxAxC).squeeze(1)

    loss = ce_crit(g_mu, predict_gnn_target)
    return loss


@register_solver("cosda")
class COSDASolver(BaseSolver):
    def build_model(self):
        faiss_threads = configure_faiss_runtime(self.config)
        backbone_name = self.config.method.get("backbone", "resnet50")
        
        # self.num_classes already includes +1 from base_solver for OSDA setting
        self.known_class = self.num_classes - 1
        
        self.net = COSDA(
            backbone_arch=backbone_name,
            embed_feat_dim=self.config.method.embed_feat_dim,
            class_num=self.num_classes,
            mlp_width=self.config.method.mlp_width,
            mlp_depth=self.config.method.mlp_depth,
            mlp_dropout=self.config.method.mlp_dropout,
            bn_type=self.config.method.bn_type
        ).to(self.device)
        self._net_forward = self.net
        self._backbone_forward = self.net.backbone_layer
        logger.info("COSDA FAISS runtime | threads=%d", faiss_threads)
        if self.compile_enabled:
            self._net_forward = self._compile_module(self.net, "cosda.net")
            self._backbone_forward = self._compile_module(self.net.backbone_layer, "cosda.backbone")

    def _get_trainable_params(self):
        param_group = []
        for k, v in self.net.backbone_layer.named_parameters():
            param_group += [{'params': v, 'lr': self.config.method.lr * 0.1}]
        for k, v in self.net.intervener.named_parameters():
            param_group += [{'params': v, 'lr': self.config.method.lr}]
        for k, v in self.net.feat_embed_layer.named_parameters():
            param_group += [{'params': v, 'lr': self.config.method.lr}]
        for k, v in self.net.class_layer.named_parameters():
            param_group += [{'params': v, 'lr': self.config.method.lr}]
        return param_group
        
    def lr_scheduler(self, optimizer, iter_num, max_iter, gamma=10, power=0.75):
        decay = (1 + gamma * iter_num / max_iter) ** (-power)
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr0'] * decay
            param_group['weight_decay'] = self.config.method.weight_decay
            param_group['momentum'] = self.config.method.momentum
            param_group['nesterov'] = True
        return optimizer

    def initalize_memory(self):
        # We process source data for initialization
        logger.info("Initializing memory module...")
        source_len = len(self.source_loader.dataset)
        embed_dim = self.config.method.embed_feat_dim
        
        memory_source_features = torch.zeros(source_len, embed_dim).to(self.device)
        memory_source_labels = torch.zeros(source_len).long().to(self.device)
        
        begin_index = 0
        self.net.eval()
        for imgs_train, imgs_label in self.source_loader:
            images = self._to_device(imgs_train)
            label = self._to_device(imgs_label)
            bs = images.shape[0]
            index = [i for i in range(begin_index, begin_index + bs)]
            begin_index += bs
            
            with torch.no_grad():
                with self._auto_cast():
                    rois = self._backbone_forward(images)
                    features_temp, _ = self.net.feat_embed_layer(rois)
                memory_source_features[index] = features_temp
                memory_source_labels[index] = label
                
        memory_source_features = memory_source_features[:begin_index]
        memory_source_labels = memory_source_labels[:begin_index]
        logger.info("Memory module initialization has finished!")
        return memory_source_features, memory_source_labels

    def get_pseudo_label(self, new_epoch=True, test=False):
        KK = self.known_class
        dim = self.config.method.embed_feat_dim
        target_len = len(self.target_loader.dataset)
        
        embed_feat_bank = torch.zeros(target_len, dim).to(self.device)
        gt_label_bank = torch.zeros(target_len).long().to(self.device)
        pred_cls_bank = torch.zeros(target_len, KK+1).to(self.device)
        
        begin_index = 0
        self.net.eval()
        for data_t, target_t in self.target_loader:
            images = self._to_device(data_t)
            label = self._to_device(target_t)
            bs = images.shape[0]
            index = [i for i in range(begin_index, begin_index + bs)]
            begin_index += bs
            
            with torch.no_grad():
                with self._auto_cast():
                    rois = self._backbone_forward(images)
                    features_temp, _ = self.net.feat_embed_layer(rois)
                    pred_cls = self.net.class_layer(features_temp, apply_softmax=True)
                embed_feat_bank[index] = features_temp
                gt_label_bank[index] = label
                pred_cls_bank[index] = pred_cls
                
        embed_feat_bank = embed_feat_bank[:begin_index]
        gt_label_bank = gt_label_bank[:begin_index]
        pred_cls_bank = pred_cls_bank[:begin_index]
        
        embed_feat_bank = embed_feat_bank / torch.norm(embed_feat_bank, p=2, dim=1, keepdim=True)
        
        data_num = pred_cls_bank.shape[0]
        pos_topk_num = int(data_num / (KK * self.config.method.K_times))
        
        sorted_pred_cls, sorted_pred_cls_idxs = torch.sort(pred_cls_bank, dim=0, descending=True)
        pos_topk_idxs = sorted_pred_cls_idxs[:pos_topk_num, :-1].t()
        
        A_flat = pos_topk_idxs.flatten().cpu().numpy()
        mask = ~np.isin(np.array([i for i in range(begin_index)]), A_flat)
        neg_embed_feat_bank = embed_feat_bank[mask]
        
        pos_topk_idxs = pos_topk_idxs.unsqueeze(2).expand([-1, -1, dim])
        embed_feat_bank_expand = embed_feat_bank.unsqueeze(0).expand([KK, -1, -1])
        pos_feat_sample = torch.gather(embed_feat_bank_expand, 1, pos_topk_idxs)
        pos_feat_proto = torch.mean(pos_feat_sample, dim=1, keepdim=True)
        pos_feat_proto = pos_feat_proto / torch.norm(pos_feat_proto, p=2, dim=-1, keepdim=True)
        
        NUM_K = int(KK * self.config.method.V_times)
        
        faiss_kmeans = faiss.Kmeans(dim, NUM_K, niter=100, verbose=False, min_points_per_centroid=1, gpu=False)
        faiss_kmeans.train(neg_embed_feat_bank.cpu().numpy())
        neg_feat_proto = torch.from_numpy(faiss_kmeans.centroids).to(self.device)
        neg_feat_proto = neg_feat_proto / torch.norm(neg_feat_proto, p=2, dim=-1, keepdim=True)
        
        all_proto = torch.cat([pos_feat_proto, neg_feat_proto.unsqueeze(1)], dim=0)
        return all_proto, neg_feat_proto, KK + neg_feat_proto.shape[0]

    def get_pseudo_label_batch(self, feature_batch, all_proto, KK):
        # Infer hard labels locally for a target batch
        psd_label_prior_simi = torch.einsum("nd, cd -> nc", feature_batch, all_proto.squeeze(1))
        psd_label_prior_idxs = torch.max(psd_label_prior_simi, dim=-1, keepdim=True)[1].squeeze(1)
        return psd_label_prior_idxs

    def forward_BETA_ce(self, v, y, intervention, int_v, int_y, target, domain='source'):
        loss_beta_dict = {}
        if domain == 'source':
            nll_1 = ce_criterion(self.num_classes, self.config, target, y, domain)
            int_nll_1 = -ce_criterion(self.num_classes, self.config, target, int_y, domain)
            loss_beta_dict['loss_kl_nll'] = self.config.method.lambda_kl * v
            loss_beta_dict['loss_kl_int_nll'] = self.config.method.lambda_kl * int_v
            
            loss_sta_s = intervention_loss(intervention, self.config.method.int_epsilon).mean()
            loss_beta_dict['loss_sta_s'] = self.config.method.lambda_sta * loss_sta_s
            loss_beta_dict['loss_e_s'] = self.config.method.lambda_beta_e * ((nll_1 + self.config.method.lambda_int * int_nll_1).mean())
        else:
            nll_1 = ce_criterion(self.num_classes, self.config, target, y, domain)
            int_nll_1 = -ce_criterion(self.num_classes, self.config, target, int_y, domain)
            loss_sta_t = intervention_loss(intervention, self.config.method.int_epsilon).mean()
            loss_beta_dict['loss_sta_t'] = self.config.method.lambda_sta * loss_sta_t
            loss_beta_dict['loss_kl_nll'] = self.config.method.lambda_kl * v
            loss_beta_dict['loss_kl_int_nll'] = self.config.method.lambda_kl * int_v
            loss_beta_dict['loss_d_t'] = self.config.method.lambda_beta_d * ((nll_1 + self.config.method.lambda_int * int_nll_1).mean())

        return loss_beta_dict

    def forward_EXO(self, memory_source_features, memory_source_labels, mean_unk_proto, hard_label_bank, target_s, train_s, features_target, NUM_K):
        self.net.train()
        loss_exo_dict = {}
        
        batch_size = int(self.config.batch_size)
        memory_source_features = memory_source_features[batch_size:]
        memory_source_labels = memory_source_labels[batch_size:]
        
        with torch.no_grad():
            rois = self._backbone_forward(train_s)
            features_temp, _ = self.net.feat_embed_layer(rois)
            
        memory_source_features = torch.cat((memory_source_features, features_temp), dim=0)
        memory_source_labels = torch.cat((memory_source_labels, target_s), dim=0)
        
        mean_source = CalculateMean(memory_source_features, memory_source_labels, self.known_class)
        mean_all = torch.cat((mean_source, mean_unk_proto), dim=0)
        
        trans_loss = MO(mean_all, features_target, hard_label_bank, NUM_K)
        loss_exo_dict['loss_cl'] = self.config.method.lambda_exo * trans_loss
        return memory_source_features, memory_source_labels, loss_exo_dict


    def extra_training_state_dict(self):
        state = super().extra_training_state_dict()
        if getattr(self, "_memory_source_features", None) is not None:
            state["memory_source_features"] = self._memory_source_features
            state["memory_source_labels"] = self._memory_source_labels
        return state

    def load_extra_training_state_dict(self, state):
        super().load_extra_training_state_dict(state)
        features = state.get("memory_source_features")
        labels = state.get("memory_source_labels")
        self._memory_source_features = (
            features.to(self.device) if features is not None else None
        )
        self._memory_source_labels = (
            labels.to(self.device) if labels is not None else None
        )

    def train(self):
        optimizer = optim.SGD(self._get_trainable_params())
        # op_copy logic
        for param_group in optimizer.param_groups:
            param_group['lr0'] = param_group['lr']

        logger.info("Start COSDASolver training...")
        
        max_epochs = self.config.method.epochs
        len_source = len(self.source_loader)
        len_target = len(self.target_loader)
        max_len = max(len_source, len_target)
        
        warm_up_epoch = int(self.config.method.warm_up_epoch)
        memory_source_features = getattr(self, "_memory_source_features", None)
        memory_source_labels = getattr(self, "_memory_source_labels", None)
        if self._resume_epoch >= warm_up_epoch:
            if memory_source_features is None:
                memory_source_features, memory_source_labels = self.initalize_memory()
                self._memory_source_features = memory_source_features
                self._memory_source_labels = memory_source_labels
            if self._resume_epoch == warm_up_epoch:
                self._pending_training_state.pop("optimizer", None)
        self.register_training_state(optimizer=optimizer)
        best_acc = self._best_metric
        
        for epoch in self._epoch_range(max_epochs):
            if epoch == warm_up_epoch and self._resume_epoch < warm_up_epoch:
                optimizer = optim.SGD(self._get_trainable_params())
                for param_group in optimizer.param_groups:
                    param_group['lr0'] = param_group['lr']
                self.register_training_state(optimizer=optimizer)
                memory_source_features, memory_source_labels = self.initalize_memory()
                self._memory_source_features = memory_source_features
                self._memory_source_labels = memory_source_labels
                    
            if epoch >= warm_up_epoch:
                all_proto, neg_proto, NUM_K = self.get_pseudo_label(new_epoch=True)
            self.net.train()
            total_loss_meter = AverageMeter()
            
            src_iter = cycle(self.source_loader)
            tgt_iter = cycle(self.target_loader)
            
            for batch_idx in range(max_len):
                train_s, target_s = next(src_iter)
                train_t, target_t = next(tgt_iter)
                
                train_s = self._to_device(train_s)
                target_s = self._to_device(target_s)
                train_t = self._to_device(train_t)
                target_t = self._to_device(target_t)
                
                self._zero_grad(optimizer)
                
                if epoch < warm_up_epoch:
                    current_stage_epoch = epoch
                    max_stage_epochs = warm_up_epoch
                else:
                    current_stage_epoch = epoch - warm_up_epoch
                    max_stage_epochs = self.config.method.epochs - warm_up_epoch
                    
                iter_idx = current_stage_epoch * max_len + batch_idx
                max_iter = max_stage_epochs * max_len
                self.lr_scheduler(optimizer, iter_idx, max_iter)
                
                loss_dict = {}
                kld_weight = kl_anneal_function(current_stage_epoch, max_len, batch_idx)
                self.config.method.lambda_kl = kld_weight
                
                # Source path
                with self._auto_cast():
                    rois_s, v_s, rois_c_s, y_s, intervention_s, int_rois_s, int_v_s, int_y_s = self._net_forward(train_s, apply_softmax=False)
                    loss_beta_s = self.forward_BETA_ce(v_s, y_s, intervention_s, int_v_s, int_y_s, target_s, domain='source')
                    loss_dict.update(loss_beta_s)
                
                if epoch >= warm_up_epoch:
                    # Target path
                    self.net.train()
                    with self._auto_cast():
                        rois_t, v_t, rois_c_t, y_t, intervention_t, int_rois_t, int_v_t, int_y_t = self._net_forward(train_t, apply_softmax=False)
                        hard_label_bank = self.get_pseudo_label_batch(rois_c_t, all_proto, self.known_class)
                        hard_label_bank[hard_label_bank >= self.known_class] = self.known_class
                        
                        loss_beta_t = self.forward_BETA_ce(v_t, y_t, intervention_t, int_v_t, int_y_t, hard_label_bank, domain='target')
                    
                    target_score, predict_target = torch.max(y_t.softmax(-1), 1)
                    idx_pseudo1 = target_score > self.config.method.confidence_th
                    idx_pseudo2 = predict_target == hard_label_bank
                    combined_mask = idx_pseudo1 & idx_pseudo2
                    
                    hard_label_bank_masked = hard_label_bank[combined_mask]
                    features_target_masked = rois_c_t[combined_mask]
                    
                    mask_label = (hard_label_bank_masked < self.known_class)
                    hard_label_bank1 = hard_label_bank_masked[mask_label]
                    features_target1 = features_target_masked[mask_label]
                    
                    if features_target1.shape[0] > 0:
                        memory_source_features, memory_source_labels, loss_align_t = self.forward_EXO(
                            memory_source_features, memory_source_labels, neg_proto, hard_label_bank1, 
                            target_s, train_s, features_target1, NUM_K
                        )
                        self._memory_source_features = memory_source_features
                        self._memory_source_labels = memory_source_labels
                        loss_dict.update(loss_align_t)
                    
                    loss_dict.update(loss_beta_t)
                
                loss_all = sum(loss for loss in loss_dict.values())
                self._optimizer_step_with_optional_clip(loss_all, optimizer)
                
                total_loss_meter.update(loss_all.item())
                
            acc = self.evaluate()
            if acc > best_acc:
                best_acc = acc
            self._maybe_save_best(acc, epoch + 1)
            self._log_epoch_summary(
                epoch + 1,
                max_epochs,
                metrics={"loss": total_loss_meter.avg},
                score=acc,
                best_score=best_acc,
                score_name="Score",
            )
        if self._load_best_checkpoint_if_available():
            self._log_best_checkpoint_loaded("Score")
        self._log_training_complete(best_score=best_acc, score_name="Score")

    def _set_train_mode(self):
        self.net.train()

    def _set_eval_mode(self):
        self.net.eval()

    def forward_for_eval(self, imgs):
        _, _, _, y, _, _, _, _ = self._net_forward(imgs, apply_softmax=False)
        return y

    def predict_with_rejection(self, preds: torch.Tensor, probs: torch.Tensor):
        return preds
