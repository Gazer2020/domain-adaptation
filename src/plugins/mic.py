"""
MIC (Masked Image Consistency) Plugin for domain adaptation.

This plugin implements the masked image consistency training strategy,
where a student model learns to predict on masked images to match
the teacher model's predictions on full images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MICPlugin(nn.Module):
    """
    Masked Image Consistency plugin.
    
    Generates random patch masks and computes consistency loss between
    student predictions on masked images and teacher predictions on full images.
    """
    
    def __init__(
        self,
        mask_ratio: float = 0.6,
        patch_size: int = 32,
        apply_to_batch: bool = True
    ):
        """
        Initialize MIC plugin.
        
        Args:
            mask_ratio: Ratio of patches to mask (default: 0.6, mask 60%)
            patch_size: Size of each patch in pixels (e.g., 32x32)
            apply_to_batch: If True, apply same mask pattern to entire batch (faster)
        """
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.apply_to_batch = apply_to_batch

    def _generate_mask(self, img: torch.Tensor) -> torch.Tensor:
        """
        Generate grid mask for images.
        
        Args:
            img: Input images of shape (B, C, H, W)
            
        Returns:
            Binary mask of shape (B, 1, H, W), 0=masked, 1=keep
        """
        B, C, H, W = img.shape
        
        # Calculate number of patches
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        num_patches = h_patches * w_patches

        # Number of patches to keep
        num_keep = int(num_patches * (1 - self.mask_ratio))

        # Generate random noise for sorting
        if self.apply_to_batch:
            noise = torch.rand(1, num_patches, device=img.device)
            noise = noise.repeat(B, 1)
        else:
            noise = torch.rand(B, num_patches, device=img.device)

        # Sort and select patches to keep
        ids_shuffle = torch.argsort(noise, dim=1)

        # Generate binary mask: 0=mask, 1=keep
        mask = torch.zeros([B, num_patches], device=img.device)
        mask.scatter_(1, ids_shuffle[:, :num_keep], 1)

        # Reshape mask to image size
        mask = mask.view(B, 1, h_patches, w_patches)
        mask = F.interpolate(mask, scale_factor=self.patch_size, mode="nearest")

        return mask

    def forward(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        target_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MIC consistency loss.
        
        Args:
            student_model: Student model being trained
            teacher_model: Teacher model (EMA or frozen)
            target_images: Target domain images
            
        Returns:
            Consistency loss between student and teacher predictions
        """
        # Teacher generates pseudo-labels using full images
        with torch.no_grad():
            teacher_logits = teacher_model(target_images)
            pseudo_labels = torch.softmax(teacher_logits, dim=1).argmax(dim=1)

        # Generate mask and apply to images
        mask = self._generate_mask(target_images)
        masked_images = target_images * mask

        # Student predicts on masked images
        student_logits = student_model(masked_images)

        # Consistency loss: student should match teacher's predictions
        loss = F.cross_entropy(student_logits, pseudo_labels)

        return loss
