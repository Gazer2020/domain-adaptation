#!/bin/bash
# Run MIC-SimSiam + PCOSL on all 6 Office-31 OSDA tasks
# This script runs sequentially and saves results

set -e

cd /home/gazer/domain-adaptation/src

# Office-31 OSDA tasks
TASKS=(
    "amazon:dslr"
    "amazon:webcam"
    "dslr:amazon"
    "dslr:webcam"
    "webcam:amazon"
    "webcam:dslr"
)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="/tmp/pcosl_results_${TIMESTAMP}.txt"
echo "PCOSL (MIC-SimSiam) OSDA Results on Office-31" > $RESULTS_FILE
echo "=============================================" >> $RESULTS_FILE
echo "Date: $(date)" >> $RESULTS_FILE
echo "Settings: unknown_ratio=0.35, lambda_contrastive=0.1" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

for task in "${TASKS[@]}"; do
    IFS=':' read -r source target <<< "$task"
    exp_name="pcosl_${source:0:1}2${target:0:1}"
    
    echo ""
    echo "=========================================="
    echo "Running: $source -> $target ($exp_name)"
    echo "=========================================="
    echo "" >> $RESULTS_FILE
    echo "Task: $source -> $target" >> $RESULTS_FILE
    
    # Run training with PCOSL settings and capture output
    uv run python main.py \
        method=mic_simsiam \
        dataset=office-31 \
        dataset.source=$source \
        dataset.target=$target \
        exp_name=$exp_name \
        method.unknown_ratio=0.35 \
        method.lambda_contrastive=0.1 \
        method.contrastive_margin=0.5 \
        2>&1 | tee /tmp/pcosl_${exp_name}.log
    
    # Extract best H-score from log
    best_hscore=$(grep -oP 'Best H-score: \K[0-9.]+' /tmp/pcosl_${exp_name}.log 2>/dev/null || echo "N/A")
    if [ "$best_hscore" == "N/A" ]; then
        # Try alternate format
        best_hscore=$(grep "best:" /tmp/pcosl_${exp_name}.log | tail -1 | grep -oP 'best: \K[0-9.]+' || echo "N/A")
    fi
    echo "Best H-score: ${best_hscore}%" >> $RESULTS_FILE
    echo "----------------------------------------" >> $RESULTS_FILE
done

echo "" >> $RESULTS_FILE
echo "=============================================" >> $RESULTS_FILE
echo "Evaluation complete!" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

echo ""
echo "=========================================="
echo "Summary of Results"
echo "=========================================="
cat $RESULTS_FILE

