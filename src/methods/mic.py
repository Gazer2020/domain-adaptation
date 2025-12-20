import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.backbones import get_resnet50
from utils import AverageMeter

logger = logging.getLogger(__name__)


class MaskSolver:
    """
    Mask solver class implementing source only training.
    Mainly used for smoke testing.
    """

    def __init__(self, config, loaders):
        """
        Initialize the MaskSolver.

        Args:
            config: OmegaConf configuration object
            loaders: Tuple containing (source_loader, target_loader, target_test_loader)
        """
        self.config = config
        self.source_loader, self.target_loader, self.target_test_loader = loaders
        self.device = torch.device(config.device)

        if self.config.method.setting == "csda":
            self.num_classes = self.config.dataset.num_classes
        else:
            raise NotImplementedError("BaseSolver only supports csda setting.")

        self.build_model()

        # Optimizer
        self.build_optimizer()

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        self.mic_module = MICPlugin(mask_ratio=0.5, patch_size=32).to(self.device)

    def build_model(self):
        """
        Build the network architecture.
        """
        stu_model = get_resnet50()
        stu_model.fc = nn.Linear(stu_model.fc.in_features, self.num_classes)

        tea_model = get_resnet50()
        tea_model.fc = nn.Linear(tea_model.fc.in_features, self.num_classes)

        self.stu_model = stu_model.to(self.device)
        self.stu_model.compile()
        self.tea_model = tea_model.to(self.device)
        self.tea_model.compile()

    def build_optimizer(self):
        """
        Build the optimizer.
        """
        # Simple SGD or Adam
        lr = self.config.method.lr
        self.optimizer = optim.SGD(
            self.stu_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
        )

    def train(self):
        """
        Main training loop.
        """
        max_epochs = self.config.method.epochs

        def cycle(iterable):
            while True:
                for x in iterable:
                    yield x

        logger.info(f"Start training for {max_epochs} epochs...")

        for epoch in range(max_epochs):
            self.stu_model.train()
            self.tea_model.train()

            tgt_iter = cycle(self.target_loader)

            sem_loss_meter = AverageMeter()
            mic_loss_meter = AverageMeter()
            tot_loss_meter = AverageMeter()

            pbar = tqdm(self.source_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
            for src_imgs, src_labels in pbar:
                tgt_imgs, _ = next(tgt_iter)

                src_imgs = src_imgs.to(self.device)
                src_labels = src_labels.to(self.device)
                tgt_imgs = tgt_imgs.to(self.device)

                self.optimizer.zero_grad()

                # Forward & Loss calculation
                src_pred = self.stu_model(src_imgs)
                sem_loss = self.criterion(src_pred, src_labels)

                mic_loss = self.mic_module(self.stu_model, self.tea_model, tgt_imgs)

                loss = sem_loss + self.config.method.lambda_mic * mic_loss

                loss.backward()
                self.optimizer.step()

                # Update teacher model
                with torch.no_grad():
                    m = self.config.method.momentum
                    for param_q, param_k in zip(
                        self.stu_model.parameters(), self.tea_model.parameters()
                    ):
                        param_k.data.mul_(m).add_((1 - m) * param_q.data)

                sem_loss_meter.update(sem_loss.item())
                mic_loss_meter.update(mic_loss.item())
                tot_loss_meter.update(loss.item())
                pbar.set_postfix(
                    {"sem_loss": sem_loss_meter.avg, "mic_loss": mic_loss_meter.avg, "tot_loss": tot_loss_meter.avg}
                )

            # Evaluation after each epoch
            acc = self.evaluate()
            logger.info(
                f"Epoch {epoch+1} finished. Avg Loss: {tot_loss_meter.avg:.4f}, Target Acc: {acc:.2f}%"
            )

        logger.info("Training finished.")

    def evaluate(self):
        """
        Evaluate on target test set.
        """
        self.stu_model.eval()

        correct_meter = AverageMeter()

        with torch.no_grad():
            for imgs, labels in self.target_test_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.stu_model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                
                correct_meter.update((predicted == labels).sum().item())

        acc = 100 * correct_meter.avg
        return acc

    def save_checkpoint(self, path):
        """
        Save model checkpoint.
        """
        torch.save(self.stu_model.state_dict(), path)

        logger.info(f"Model saved to {path}")


import torch
import torch.nn as nn
import torch.nn.functional as F


class MICPlugin(nn.Module):
    def __init__(self, mask_ratio=0.6, patch_size=32, apply_to_batch=True):
        """
        MIC 插件初始化
        :param mask_ratio: 遮挡比例 (默认 0.6，即遮挡 60% 的区域)
        :param patch_size: 遮挡块的大小 (像素)，例如 32x32 的块
        :param apply_to_batch: 是否对整个 batch 应用相同的 mask pattern (效率更高) 或每张图独立 mask
        """
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.apply_to_batch = apply_to_batch

    def _generate_mask(self, img):
        """生成网格掩码"""
        B, C, H, W = img.shape
        # 计算网格数量
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        num_patches = h_patches * w_patches

        # 计算需要保留的 patch 数量
        num_keep = int(num_patches * (1 - self.mask_ratio))

        # 生成随机掩码索引
        # 如果 apply_to_batch=True，则所有图片共用一个 mask 模式 (速度快)
        # 否则每张图片单独生成 mask
        if self.apply_to_batch:
            noise = torch.rand(1, num_patches, device=img.device)
            noise = noise.repeat(B, 1)
        else:
            noise = torch.rand(B, num_patches, device=img.device)

        # 排序并选取要保留的索引
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 生成 binary mask: 0 表示遮挡, 1 表示保留
        mask = torch.zeros([B, num_patches], device=img.device)
        mask.scatter_(1, ids_shuffle[:, :num_keep], 1)

        # 将 mask 还原回 (B, 1, H, W) 的图像尺寸
        mask = mask.view(B, 1, h_patches, w_patches)
        mask = F.interpolate(mask, scale_factor=self.patch_size, mode="nearest")

        return mask

    def forward(self, student_model, teacher_model, target_images):
        """
        前向传播并计算 MIC Loss
        :param student_model: 正在更新的学生模型
        :param teacher_model: 冻结的/EMA的教师模型 (提供伪标签)
        :param target_images: 原始目标域图片
        :return: loss, correct_count (用于监控)
        """

        # 1. Teacher 生成伪标签 (使用全图)
        with torch.no_grad():
            teacher_logits = teacher_model(target_images)
            # 获取 Teacher 的预测类别作为伪标签
            pseudo_label = torch.softmax(teacher_logits, dim=1).argmax(dim=1)
            # 可选：可以在这里加入阈值过滤 (confidence thresholding)

        # 2. 生成掩码并应用到图片
        mask = self._generate_mask(target_images)
        masked_images = target_images * mask  # 广播机制应用掩码

        # 3. Student 进行预测 (使用残图)
        student_logits = student_model(masked_images)

        # 4. 计算一致性损失 (Cross Entropy)
        # 强迫 Student 看残图也能预测出 Teacher 看全图的结果
        loss = F.cross_entropy(student_logits, pseudo_label)

        return loss
