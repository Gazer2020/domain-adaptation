"""
Channel Activation Visualization Script

This script validates the hypothesis that known and unknown classes have different
channel activation patterns in the last convolutional layer of a pretrained ResNet.

Usage:
    uv run python scripts/visualize_channel_acts.py \
        --checkpoint results/source_only_osda/checkpoints/source_only_osda.pth \
        --data-root data/mini-office-31 \
        --source amazon \
        --target webcam \
        --output results/visualizations/
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.manifold import TSNE
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets.loader import DomainDataset


def get_transform():
    """Standard ImageNet transform for inference."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class ChannelActivationExtractor:
    """
    Extract channel activations from the last conv layer of ResNet.
    
    Uses forward hooks to capture intermediate activations.
    """
    
    def __init__(self, model: nn.Module, layer_name: str = "layer4"):
        self.model = model
        self.model.eval()
        self.activations = None
        self.layer_name = layer_name
        
        # Register hook on the target layer
        target_layer = getattr(model, layer_name)
        target_layer.register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module, input, output):
        """Hook to capture activations."""
        # output shape: [B, C, H, W]
        # We take global average pooling to get [B, C]
        self.activations = output.mean(dim=[2, 3]).detach()
    
    @torch.no_grad()
    def extract(self, dataloader: DataLoader, device: torch.device):
        """
        Extract channel activations for all samples in the dataloader.
        
        Returns:
            activations: numpy array of shape [N, C]
            labels: numpy array of shape [N]
        """
        all_activations = []
        all_labels = []
        
        for imgs, labels in tqdm(dataloader, desc="Extracting activations"):
            imgs = imgs.to(device)
            _ = self.model(imgs)  # Forward pass triggers the hook
            
            all_activations.append(self.activations.cpu().numpy())
            all_labels.append(labels.numpy())
        
        return np.concatenate(all_activations), np.concatenate(all_labels)


def load_model(checkpoint_path: Path, num_classes: int, backbone: str = "resnet50"):
    """Load a trained ResNet model from checkpoint."""
    if backbone == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
    elif backbone == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
    else:
        model = models.resnet101(weights=None)
        in_features = model.fc.in_features
    
    model.fc = nn.Linear(in_features, num_classes)
    
    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: checkpoint {checkpoint_path} not found, using random weights")
    
    return model


def visualize_channel_distributions(
    known_acts: np.ndarray,
    unknown_acts: np.ndarray,
    output_dir: Path,
    top_k: int = 20
):
    """
    Visualize channel activation distributions for known vs unknown classes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute mean activation per channel
    known_mean = known_acts.mean(axis=0)
    unknown_mean = unknown_acts.mean(axis=0)
    known_std = known_acts.std(axis=0)
    unknown_std = unknown_acts.std(axis=0)
    
    num_channels = known_mean.shape[0]
    
    # 1. Bar plot comparing mean activations
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(num_channels)
    width = 0.35
    ax.bar(x - width/2, known_mean, width, label='Known', alpha=0.7, color='blue')
    ax.bar(x + width/2, unknown_mean, width, label='Unknown', alpha=0.7, color='red')
    ax.set_xlabel('Channel Index')
    ax.set_ylabel('Mean Activation')
    ax.set_title('Channel Activations: Known vs Unknown Classes')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "channel_mean_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: channel_mean_comparison.png")
    
    # 2. Difference plot (most discriminative channels)
    diff = np.abs(known_mean - unknown_mean)
    top_channels = np.argsort(diff)[-top_k:][::-1]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(top_k)
    ax.barh(x, diff[top_channels], color='purple', alpha=0.7)
    ax.set_yticks(x)
    ax.set_yticklabels([f"Ch {c}" for c in top_channels])
    ax.set_xlabel('Absolute Difference in Mean Activation')
    ax.set_title(f'Top {top_k} Most Discriminative Channels')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "top_discriminative_channels.png", dpi=150)
    plt.close()
    print(f"Saved: top_discriminative_channels.png")
    
    # 3. Histogram for top discriminative channels
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for i, ch in enumerate(top_channels[:10]):
        ax = axes[i]
        ax.hist(known_acts[:, ch], bins=30, alpha=0.6, label='Known', color='blue', density=True)
        ax.hist(unknown_acts[:, ch], bins=30, alpha=0.6, label='Unknown', color='red', density=True)
        ax.set_title(f'Channel {ch}')
        ax.legend(fontsize=8)
    plt.suptitle('Activation Distributions for Top 10 Discriminative Channels')
    plt.tight_layout()
    plt.savefig(output_dir / "channel_histograms.png", dpi=150)
    plt.close()
    print(f"Saved: channel_histograms.png")
    
    # 4. Statistics summary
    stats = {
        "num_channels": num_channels,
        "num_known_samples": len(known_acts),
        "num_unknown_samples": len(unknown_acts),
        "top_discriminative_channels": top_channels.tolist(),
        "mean_diff": diff[top_channels].tolist(),
    }
    
    # Compute KL-divergence-like metric for each channel
    eps = 1e-8
    kl_scores = []
    for ch in range(num_channels):
        k_hist, bins = np.histogram(known_acts[:, ch], bins=50, density=True)
        u_hist, _ = np.histogram(unknown_acts[:, ch], bins=bins, density=True)
        k_hist = k_hist + eps
        u_hist = u_hist + eps
        kl = np.sum(k_hist * np.log(k_hist / u_hist))
        kl_scores.append(kl)
    
    kl_scores = np.array(kl_scores)
    top_kl_channels = np.argsort(kl_scores)[-top_k:][::-1]
    
    # Save summary
    with open(output_dir / "statistics.txt", "w") as f:
        f.write("Channel Activation Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of channels: {num_channels}\n")
        f.write(f"Known class samples: {len(known_acts)}\n")
        f.write(f"Unknown class samples: {len(unknown_acts)}\n\n")
        f.write(f"Top {top_k} discriminative channels (by mean diff):\n")
        for i, ch in enumerate(top_channels):
            f.write(f"  {i+1}. Channel {ch}: diff={diff[ch]:.4f}\n")
        f.write(f"\nTop {top_k} discriminative channels (by KL-divergence):\n")
        for i, ch in enumerate(top_kl_channels):
            f.write(f"  {i+1}. Channel {ch}: KL={kl_scores[ch]:.4f}\n")
    print(f"Saved: statistics.txt")
    
    return stats


def visualize_tsne(
    known_acts: np.ndarray,
    unknown_acts: np.ndarray,
    output_dir: Path
):
    """
    t-SNE visualization of channel activation vectors.
    """
    # Combine data
    all_acts = np.vstack([known_acts, unknown_acts])
    labels = np.concatenate([
        np.zeros(len(known_acts)),
        np.ones(len(unknown_acts))
    ])
    
    # Subsample if too many samples
    max_samples = 2000
    if len(all_acts) > max_samples:
        indices = np.random.choice(len(all_acts), max_samples, replace=False)
        all_acts = all_acts[indices]
        labels = labels[indices]
    
    print(f"Running t-SNE on {len(all_acts)} samples...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings = tsne.fit_transform(all_acts)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    known_mask = labels == 0
    unknown_mask = labels == 1
    
    ax.scatter(
        embeddings[known_mask, 0], embeddings[known_mask, 1],
        c='blue', label='Known', alpha=0.5, s=20
    )
    ax.scatter(
        embeddings[unknown_mask, 0], embeddings[unknown_mask, 1],
        c='red', label='Unknown', alpha=0.5, s=20
    )
    ax.set_title('t-SNE of Channel Activations\n(Known vs Unknown Classes)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "tsne_channel_activations.png", dpi=150)
    plt.close()
    print(f"Saved: tsne_channel_activations.png")


def main():
    parser = argparse.ArgumentParser(description="Visualize channel activations for OSDA")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--data-root", type=str, default="data/mini-office-31", help="Dataset root")
    parser.add_argument("--source", type=str, default="amazon", help="Source domain")
    parser.add_argument("--target", type=str, default="webcam", help="Target domain")
    parser.add_argument("--output", type=str, default="results/visualizations", help="Output directory")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Backbone architecture")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # OSDA split: source has classes 0-9, target has 0-9 (known) + 20-30 (unknown)
    # For this visualization, we use:
    # - Known: classes 0-9 from target domain (shared with source)
    # - Unknown: classes 20-30 from target domain (not in source)
    known_classes = list(range(0, 10))
    unknown_classes = list(range(20, 31))
    num_source_classes = len(known_classes)
    
    data_root = Path(args.data_root)
    target_path = data_root / args.target
    
    if not target_path.exists():
        print(f"Error: Target domain path {target_path} does not exist")
        return
    
    transform = get_transform()
    
    # Create datasets for known and unknown classes
    known_dataset = DomainDataset(target_path, known_classes, transform=transform)
    unknown_dataset = DomainDataset(target_path, unknown_classes, transform=transform)
    
    print(f"Known class samples: {len(known_dataset)}")
    print(f"Unknown class samples: {len(unknown_dataset)}")
    
    known_loader = DataLoader(known_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    unknown_loader = DataLoader(unknown_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Load model
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path("nonexistent.pth")
    model = load_model(checkpoint_path, num_source_classes, args.backbone)
    model = model.to(device)
    model.eval()
    
    # Create extractor
    extractor = ChannelActivationExtractor(model, layer_name="layer4")
    
    # Extract activations
    print("\nExtracting known class activations...")
    known_acts, known_labels = extractor.extract(known_loader, device)
    
    print("\nExtracting unknown class activations...")
    unknown_acts, unknown_labels = extractor.extract(unknown_loader, device)
    
    print(f"\nKnown activations shape: {known_acts.shape}")
    print(f"Unknown activations shape: {unknown_acts.shape}")
    
    # Visualize
    output_dir = Path(args.output)
    
    print("\nGenerating visualizations...")
    stats = visualize_channel_distributions(known_acts, unknown_acts, output_dir)
    visualize_tsne(known_acts, unknown_acts, output_dir)
    
    print(f"\n✅ All visualizations saved to: {output_dir}")
    print("\nKey findings:")
    print(f"  - Top 5 discriminative channels: {stats['top_discriminative_channels'][:5]}")
    print(f"  - Check the plots to see if known/unknown have distinct patterns!")


if __name__ == "__main__":
    main()
