#!/bin/bash
# Ablation study script for TwinTrack
# Usage: bash scripts/ablation_study.sh
set -e

DATASET=${1:-DCT}

# 配置组合：backbone, TLCFS, DLTC, TEL
for BACKBONE in resnet50 resnet101; do
  for TLCFS in true false; do
    for DLTC in true false; do
      for TEL in true false; do
        EXP_NAME="${DATASET}_${BACKBONE}_tlcfs${TLCFS}_dltc${DLTC}_tel${TEL}"
        echo "\n===== Running $EXP_NAME ====="
        # 训练
        python twintrack/train.py --dataset $DATASET --backbone $BACKBONE --use_tlcfs $TLCFS --use_dltc $DLTC --use_tel $TEL --log_dir logs/$EXP_NAME --checkpoint_dir checkpoints/$EXP_NAME
        # 测试
        python twintrack/test.py --dataset $DATASET --backbone $BACKBONE --use_tlcfs $TLCFS --use_dltc $DLTC --use_tel $TEL --log_dir logs/$EXP_NAME --checkpoint_dir checkpoints/$EXP_NAME
        # 结果收集
        tail -n 5 logs/$EXP_NAME/test.log | tee -a ablation_results.txt
      done
    done
  done
done

echo "Ablation study finished. Results saved to ablation_results.txt" 