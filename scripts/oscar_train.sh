#!/bin/bash
#SBATCH -J bach
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Submit with:
#   sbatch -J bach_m1 scripts/oscar_train.sh                                   # M1 SFT
#   sbatch -J bach_m2 --export=ALL,STAGE=m2 scripts/oscar_train.sh             # M2 PPO (RL)
#   sbatch -J bach_m2dl --export=ALL,STAGE=m2_diffloss scripts/oscar_train.sh  # M2 diff-loss ablation
#   sbatch -J bach_m4 --export=ALL,STAGE=m4 scripts/oscar_train.sh             # M4 chord-conditioned

set -euo pipefail

module load anaconda3/2023.09-0-aqbc || true
module load cuda/12.9.0-cinr || true

# source activate bach_transformer

mkdir -p logs

# Unbuffered stdout/stderr so we can tail -F the log and see where Python is
# in real time. Without this, SLURM redirects stdout to a file which triggers
# block buffering — a silently-dying process flushes nothing, making crashes
# look like "job produced no output" (see job 1804127 post-mortem).
export PYTHONUNBUFFERED=1

STAGE="${STAGE:-m1}"

case "$STAGE" in
    m1)
        python -m train.train_m1 --config configs/base.yaml
        ;;
    m2)
        # PPO fine-tune from M1. init_ckpt + ckpt_dir are config defaults.
        python -m train.train_m2 --config configs/base.yaml
        ;;
    m4)
        # M4: chord-conditioned SFT. Bigger model (8 layers, d_ff=1536)
        # to handle the 25% longer interleaved sequence without diluting
        # pitch-to-pitch attention capacity.
        python -m train.train_m4 --config configs/base.yaml \
            --override chord.enabled=true \
            --override model.n_layers=8 \
            --override model.d_ff=1536
        ;;
    m2_diffloss)
        # Legacy differentiable-rule-loss baseline (ablation / comparison).
        python -m train.train_m2_diffloss --config configs/base.yaml \
            --override rule_loss.enabled=true \
            --override train.ckpt_dir=checkpoints/m2_diffloss
        ;;
    *)
        echo "Unknown STAGE=$STAGE" >&2
        exit 1
        ;;
esac