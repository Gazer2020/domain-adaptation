#!/bin/bash
# Test script for OSDA handling improvements
# This script validates that the OSDA changes work correctly

set -e

cd "$(dirname "$0")/../src"

echo "=========================================="
echo "OSDA Handling Test Suite"
echo "=========================================="

# Test 1: OSDA setting with CAD method
echo ""
echo "[Test 1] Testing OSDA setting with CAD method..."
echo "Running: uv run python main.py method=cad dataset=mini-office-31 method.setting=osda method.pretrain_epochs=1 method.adapt_epochs=1 exp_name=osda_cad_test"
echo ""

uv run python main.py \
    method=cad \
    dataset=mini-office-31 \
    method.setting=osda \
    method.pretrain_epochs=1 \
    method.adapt_epochs=1 \
    exp_name=osda_cad_test \
    batch_size=8 \
    num_workers=0

echo ""
echo "[Test 1] PASSED - OSDA with CAD completed successfully"
echo ""

# Test 2: CSDA setting for regression test
echo "[Test 2] Testing CSDA setting for regression (ROS method)..."
echo "Running: uv run python main.py method=ros dataset=mini-office-31 method.setting=csda method.epochs=1 exp_name=csda_ros_test"
echo ""

uv run python main.py \
    method=ros \
    dataset=mini-office-31 \
    method.setting=csda \
    method.epochs=1 \
    exp_name=csda_ros_test \
    batch_size=8 \
    num_workers=0

echo ""
echo "[Test 2] PASSED - CSDA regression test completed successfully"
echo ""

# Test 3: SourceOnly baseline with OSDA
echo "[Test 3] Testing SourceOnly baseline with OSDA setting..."
echo "Running: uv run python main.py method=source_only dataset=mini-office-31 method.setting=osda method.epochs=1 exp_name=osda_sourceonly_test"
echo ""

uv run python main.py \
    method=source_only \
    dataset=mini-office-31 \
    method.setting=osda \
    method.epochs=1 \
    exp_name=osda_sourceonly_test \
    batch_size=8 \
    num_workers=0

echo ""
echo "[Test 3] PASSED - SourceOnly with OSDA completed successfully"
echo ""

echo "=========================================="
echo "All tests passed!"
echo "=========================================="
