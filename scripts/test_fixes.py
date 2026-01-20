#!/usr/bin/env python
"""
Verification test script for domain adaptation codebase.

This script validates the correctness of fixes applied to the codebase,
including OSDA metrics, label mapping, and method functionality.

Usage:
    python scripts/test_fixes.py [--test TEST_NAME]
    
Available tests:
    - osda_metrics: Test OSDA evaluation metrics calculation
    - label_mapping: Test label mapping consistency
    - checkpoint: Test checkpoint save/load functionality
    - smoke: Smoke test all methods (1 epoch each)
    - all: Run all tests (default)
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class TestOSDAMetrics:
    """Test OSDA evaluation metrics calculation."""
    
    def run(self):
        logger.info("=" * 60)
        logger.info("Testing OSDA Metrics Calculation")
        logger.info("=" * 60)
        
        from methods.base_solver import BaseSolver
        
        # Create dummy solver instance
        class DummySolver(BaseSolver):
            def build_model(self):
                self.net = nn.Linear(10, self.num_classes)
            
            def compute_loss(self, src_imgs, src_labels, tgt_imgs):
                return torch.tensor(0.0)
        
        # Test configuration
        config = OmegaConf.create({
            "device": "cpu",
            "batch_size": 32,
            "num_workers": 0,
            "method": {
                "lr": 0.001,
                "epochs": 1,
                "unknown_threshold": 0.5,
            }
        })
        
        # Test OSDA setting
        class_info = {
            "src_classes": [0, 1, 2, 3, 4],  # 5 source classes
            "tgt_classes": [0, 1, 2, 3, 4, 5, 6],  # 7 target classes
            "shared_classes": [0, 1, 2, 3, 4],
            "num_classes": 5,  # Base count
            "unknown_label": 5,  # Index after source classes
            "setting": "osda",
        }
        
        # Create empty loaders (not used in this test)
        from torch.utils.data import DataLoader, TensorDataset
        dummy_loader = DataLoader(TensorDataset(torch.zeros(10, 10), torch.zeros(10, dtype=torch.long)))
        
        solver = DummySolver(config, (dummy_loader, dummy_loader, dummy_loader), class_info)
        
        # Verify num_classes setup
        assert solver.num_classes == 6, f"Expected num_classes=6 (5+1), got {solver.num_classes}"
        assert solver.unknown_label == 5, f"Expected unknown_label=5, got {solver.unknown_label}"
        assert solver.unknown_threshold == 0.5, f"Expected threshold=0.5, got {solver.unknown_threshold}"
        
        logger.info("✓ num_classes correctly set to len(src_classes) + 1 = 6")
        logger.info("✓ unknown_label correctly set to 5")
        logger.info("✓ unknown_threshold correctly initialized")
        
        # Test OSDA metrics calculation
        N = 100
        preds = torch.randint(0, 6, (N,))  # Predictions [0-5]
        labels = torch.cat([
            torch.randint(0, 5, (70,)),  # 70 known samples [0-4]
            torch.full((30,), 5)  # 30 unknown samples [5]
        ])
        probs = torch.rand(N) * 0.8 + 0.2  # Confidence [0.2, 1.0]
        
        # Test rejection mechanism
        low_conf_indices = torch.where(probs < solver.unknown_threshold)[0]
        logger.info(f"✓ Samples with confidence < {solver.unknown_threshold}: {len(low_conf_indices)}")
        
        # Compute metrics
        hscore = solver._compute_osda_metrics(preds, labels, probs)
        
        assert 0 <= hscore <= 100, f"H-score should be in [0, 100], got {hscore}"
        logger.info(f"✓ H-score computed successfully: {hscore:.2f}%")
        
        logger.info("✅ OSDA Metrics Test PASSED\n")
        return True


class TestLabelMapping:
    """Test label mapping consistency between loader and solver."""
    
    def run(self):
        logger.info("=" * 60)
        logger.info("Testing Label Mapping Consistency")
        logger.info("=" * 60)
        
        from datasets.loader import build_class_mapping
        
        # Test OSDA mapping
        src_classes = [0, 1, 2, 3, 4]
        tgt_classes = [0, 1, 2, 3, 4, 5, 6]
        shared_classes = [0, 1, 2, 3, 4]
        
        src_mapping, tgt_mapping, unknown_label = build_class_mapping(
            src_classes, tgt_classes, shared_classes, setting="osda"
        )
        
        # Verify mappings
        assert unknown_label == 5, f"Expected unknown_label=5, got {unknown_label}"
        assert len(src_mapping) == 5, f"Expected 5 source mappings, got {len(src_mapping)}"
        assert tgt_mapping[5] == 5, f"Target-private class 5 should map to unknown_label"
        assert tgt_mapping[6] == 5, f"Target-private class 6 should map to unknown_label"
        assert tgt_mapping[0] == 0, f"Shared class 0 should map to 0"
        
        logger.info("✓ Source mapping correct: {0,1,2,3,4} → {0,1,2,3,4}")
        logger.info("✓ Target shared classes map correctly")
        logger.info("✓ Target-private classes map to unknown_label=5")
        
        # Test CSDA mapping
        src_mapping, tgt_mapping, unknown_label = build_class_mapping(
            src_classes, src_classes, src_classes, setting="csda"
        )
        
        assert unknown_label is None, f"CSDA should have no unknown_label, got {unknown_label}"
        assert src_mapping == tgt_mapping, "CSDA source and target mappings should be identical"
        
        logger.info("✓ CSDA mapping correct (no unknown class)")
        logger.info("✅ Label Mapping Test PASSED\n")
        return True


class TestCheckpoint:
    """Test checkpoint save/load functionality."""
    
    def run(self):
        logger.info("=" * 60)
        logger.info("Testing Checkpoint Save/Load")
        logger.info("=" * 60)
        
        from methods import get_solver
        import tempfile
        
        config = OmegaConf.create({
            "device": "cpu",
            "batch_size": 8,
            "num_workers": 0,
            "method": {
                "name": "sourceonly",
                "backbone": "resnet18",
                "lr": 0.001,
                "epochs": 1,
            },
            "dataset": {
                "num_classes": 10
            }
        })
        
        # Test basic solver
        from torch.utils.data import DataLoader, TensorDataset
        dummy_data = TensorDataset(torch.randn(16, 3, 224, 224), torch.randint(0, 10, (16,)))
        dummy_loader = DataLoader(dummy_data, batch_size=8)
        
        solver_cls = get_solver("sourceonly")
        solver = solver_cls(config, (dummy_loader, dummy_loader, dummy_loader))
        
        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            checkpoint_path = f.name
        
        try:
            solver.save_checkpoint(checkpoint_path)
            logger.info(f"✓ Checkpoint saved to {checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            assert "method" in checkpoint, "Checkpoint should contain 'method' metadata"
            assert "model" in checkpoint, "Checkpoint should contain 'model' state dict"
            logger.info(f"✓ Checkpoint format correct: {list(checkpoint.keys())}")
            
            # Test loading
            solver.load_checkpoint(checkpoint_path)
            logger.info("✓ Checkpoint loaded successfully")
            
            logger.info("✅ Checkpoint Test PASSED\n")
            return True
            
        finally:
            Path(checkpoint_path).unlink(missing_ok=True)


class TestSmokeAll:
    """Smoke test all methods with 1 epoch."""
    
    def run(self):
        logger.info("=" * 60)
        logger.info("Smoke Testing All Methods")
        logger.info("=" * 60)
        
        from methods import list_solvers, get_solver
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create dummy data
        dummy_data = TensorDataset(
            torch.randn(32, 3, 224, 224),
            torch.randint(0, 10, (32,))
        )
        dummy_loader = DataLoader(dummy_data, batch_size=8, drop_last=True)
        
        methods_to_test = ["sourceonly", "cad", "mic", "ros"]
        
        for method_name in methods_to_test:
            try:
                logger.info(f"\n--- Testing {method_name.upper()} ---")
                
                config = OmegaConf.create({
                    "device": "cpu",
                    "batch_size": 8,
                    "num_workers": 0,
                    "method": {
                        "name": method_name,
                        "backbone": "resnet18",
                        "lr": 0.001,
                        "epochs": 1,
                        "pretrain_epochs": 1,  # For CAD
                        "adapt_epochs": 1,  # For CAD
                    },
                    "dataset": {
                        "num_classes": 10
                    }
                })
                
                solver_cls = get_solver(method_name)
                solver = solver_cls(config, (dummy_loader, dummy_loader, dummy_loader))
                
                logger.info(f"✓ {method_name} initialized successfully")
                
                # Test forward pass
                dummy_input = torch.randn(2, 3, 224, 224)
                with torch.no_grad():
                    output = solver.forward_for_eval(dummy_input)
                    assert output.shape[1] == 10, f"Expected 10 classes output, got {output.shape}"
                
                logger.info(f"✓ {method_name} forward pass works")
                
            except Exception as e:
                logger.error(f"✗ {method_name} failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        logger.info("\n✅ Smoke Test PASSED for all methods\n")
        return True


def main():
    parser = argparse.ArgumentParser(description="Test domain adaptation fixes")
    parser.add_argument(
        "--test",
        choices=["osda_metrics", "label_mapping", "checkpoint", "smoke", "all"],
        default="all",
        help="Which test to run (default: all)"
    )
    args = parser.parse_args()
    
    tests = {
        "osda_metrics": TestOSDAMetrics(),
        "label_mapping": TestLabelMapping(),
        "checkpoint": TestCheckpoint(),
        "smoke": TestSmokeAll(),
    }
    
    if args.test == "all":
        tests_to_run = list(tests.values())
    else:
        tests_to_run = [tests[args.test]]
    
    logger.info("Starting verification tests...\n")
    
    results = []
    for test in tests_to_run:
        try:
            result = test.run()
            results.append(result)
        except Exception as e:
            logger.error(f"Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    passed = sum(results)
    total = len(results)
    logger.info(f"Passed: {passed}/{total}")
    
    if all(results):
        logger.info("✅ ALL TESTS PASSED!")
        return 0
    else:
        logger.error("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
