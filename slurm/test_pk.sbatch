#!/bin/bash

#SBATCH --job-name=clax_pk_tests
#SBATCH -o /lustre/work/n2minh/std/clax/test_pk.out.%j
#SBATCH -e /lustre/work/n2minh/std/clax/test_pk.err.%j
#SBATCH --mail-type=all
#SBATCH --mail-user=nhat.minh.nguyen@ipmu.jp
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --nodelist=igpu01,igpu02,igpu03,igpu04,igpu05,igpu06,igpu07,igpu08

set -euo pipefail

ENV_NAME="${ENV_NAME:-clax}"
MODE="${MODE:-fast}"        # fast | full | all
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

micromamba activate "${ENV_NAME}"

export OMP_NUM_THREADS="${THREADS}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

cd /lustre/work/n2minh/clax

echo "=========================================="
echo "clax P(k) test suite"
echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Mode:   ${MODE}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=========================================="

python -u scripts/run_pk_tests.py "--${MODE}" -v
