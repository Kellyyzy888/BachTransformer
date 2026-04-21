#!/usr/bin/env bash
#
# One-shot OSCAR setup: pushes the project to OSCAR, creates a conda env,
# installs dependencies, and verifies the GPU + dataset are visible.
#
# Run from the repo root on your *local* machine:
#     bash scripts/setup_oscar.sh
#
# Prereqs:
#   - You can `ssh $OSCAR_USER@ssh.ccv.brown.edu` without a password prompt
#     (set up an SSH key at https://docs.ccv.brown.edu/oscar/access/ssh-key).
#   - rsync is installed locally (it is, on macOS and most Linuxes).
#   - The JSB-Chorales-dataset/ folder exists locally (task #10 — done).
#
# Override any of these on the command line:
#     OSCAR_USER=jdoe REMOTE_DIR=/users/jdoe/projects/bach bash scripts/setup_oscar.sh

set -euo pipefail

# ---------- config (override via env vars) ---------------------------------
OSCAR_USER="${OSCAR_USER:-$USER}"
OSCAR_HOST="${OSCAR_HOST:-ssh.ccv.brown.edu}"
REMOTE_DIR="${REMOTE_DIR:-/users/${OSCAR_USER}/bach_transformer}"
ENV_NAME="${ENV_NAME:-bach_transformer}"
PY_VERSION="${PY_VERSION:-3.10}"
ANACONDA_MODULE="${ANACONDA_MODULE:-anaconda3/2023.09-0-aqbc}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.9.0-cinr}"

SSH_TARGET="${OSCAR_USER}@${OSCAR_HOST}"

# ---------- pretty printing ------------------------------------------------
GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; RESET="\033[0m"
say()  { printf "${GREEN}==>${RESET} %s\n" "$*"; }
warn() { printf "${YELLOW}==>${RESET} %s\n" "$*"; }
fail() { printf "${RED}==>${RESET} %s\n" "$*" >&2; exit 1; }

# ---------- 1. local sanity checks ----------------------------------------
say "Local sanity checks..."

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

[[ -f requirements.txt ]]                   || fail "no requirements.txt — run from repo root"
[[ -d JSB-Chorales-dataset ]]               || fail "JSB-Chorales-dataset/ missing — finish task #10 first"
[[ -f JSB-Chorales-dataset/jsb-chorales-16th.pkl ]] \
                                             || fail "jsb-chorales-16th.pkl missing inside JSB-Chorales-dataset/"
command -v rsync >/dev/null                 || fail "rsync not installed locally"
command -v ssh   >/dev/null                 || fail "ssh not installed locally"

say "  user=${OSCAR_USER}  host=${OSCAR_HOST}  remote_dir=${REMOTE_DIR}  env=${ENV_NAME}"

# ---------- 2. ssh reachability -------------------------------------------
# Note: OSCAR requires Duo 2FA on the *first* connection even with a key, so we
# don't use BatchMode=yes here. If you have ControlMaster configured in
# ~/.ssh/config, this Duo-prompts once and every later ssh/rsync call reuses
# the open channel silently.
say "Probing SSH to ${SSH_TARGET} (Duo may prompt once)..."
if ! ssh -o ConnectTimeout=15 "$SSH_TARGET" "echo ok" >/dev/null; then
    fail "SSH to ${SSH_TARGET} failed. Check your key is uploaded and Duo was approved."
fi
say "  ssh OK"

# ---------- 3. push code (excluding heavy/junk paths) ---------------------
say "Pushing code to ${REMOTE_DIR} via rsync..."
ssh "$SSH_TARGET" "mkdir -p '${REMOTE_DIR}'"
rsync -avz --delete \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'checkpoints/' \
    --exclude 'samples/' \
    --exclude 'logs/' \
    --exclude '.DS_Store' \
    "$REPO_ROOT"/ "${SSH_TARGET}:${REMOTE_DIR}/"
say "  rsync done"

# ---------- 4. remote env setup -------------------------------------------
say "Creating conda env on remote (this may take 5-10 min)..."

ssh "$SSH_TARGET" bash -s <<REMOTE_EOF
set -euo pipefail

cd "${REMOTE_DIR}"

module load "${ANACONDA_MODULE}" || { echo "module load ${ANACONDA_MODULE} failed"; exit 1; }
module load "${CUDA_MODULE}"     || echo "(warning) module load ${CUDA_MODULE} failed — torch may fall back to CPU"

# create env if it doesn't exist
if conda env list | awk '{print \$1}' | grep -qx "${ENV_NAME}"; then
    echo "==> conda env '${ENV_NAME}' already exists — reusing"
else
    echo "==> creating conda env '${ENV_NAME}' (python=${PY_VERSION})"
    conda create -n "${ENV_NAME}" "python=${PY_VERSION}" -y
fi

source activate "${ENV_NAME}"

echo "==> upgrading pip + installing requirements"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

echo "==> verifying torch + CUDA"
python - <<PY
import torch
print(f"torch     = {torch.__version__}")
print(f"cuda      = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device    = {torch.cuda.get_device_name(0)}")
else:
    print("(CUDA not visible from login node — this is expected; the GPU comes online at job-submit time)")
PY

echo "==> verifying JSB pickle is reachable"
python -m data.jsb_loader --inspect JSB-Chorales-dataset/jsb-chorales-16th.pkl

echo "==> verifying rule_checker shim"
python -c "from eval.rule_checker import score_midi; print('rule_checker.score_midi OK')"

echo "==> setup complete"
REMOTE_EOF

# ---------- 5. wrap up ----------------------------------------------------
say ""
say "Done. To launch training:"
echo "    ssh ${SSH_TARGET}"
echo "    cd ${REMOTE_DIR}"
echo "    sbatch scripts/oscar_train.sh                       # M1 baseline"
echo "    sbatch --export=ALL,STAGE=m2 scripts/oscar_train.sh # M2 rule-aware"
echo ""
echo "Watch with:"
echo "    ssh ${SSH_TARGET} 'squeue -u ${OSCAR_USER}'"
echo "    ssh ${SSH_TARGET} 'tail -f ${REMOTE_DIR}/logs/bach_m1_*.out'"
