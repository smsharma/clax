#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --account=phy230064p
#SBATCH --job-name=claude-loop
#SBATCH --output=/ocean/projects/phy230064p/smishrasharma/jaxclass/loop_logs/slurm_%j.out
#
# Autonomous Claude loop for jaxCLASS (Carlini pattern).
# Submit: sbatch scripts/gpu_claude_loop.sh
# Monitor: tail -f loop_logs/slurm_*.out
#          ls -lt loop_logs/agent_*.log | head

cd /ocean/projects/phy230064p/smishrasharma/jaxclass

# Setup environment
set +eu
source /opt/packages/anaconda3-2022.10/etc/profile.d/conda.sh
conda activate /ocean/projects/phy230064p/smishrasharma/.conda/envs/jaxclass
set -eu

export PATH="$HOME/.local/bin:$PATH"
source "$HOME/.env"
export CI=true

export JAX_COMPILATION_CACHE_DIR="${LOCAL:-/tmp}/jax_cache"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

git config user.email "smsharma@mit.edu"
git config user.name "Siddharth Mishra-Sharma"

mkdir -p loop_logs

ITER=0
while true; do
    ITER=$((ITER + 1))
    COMMIT=$(git rev-parse --short=6 HEAD)
    LOGFILE="loop_logs/agent_${COMMIT}_$(date '+%Y%m%d_%H%M%S').log"

    echo "=============================================="
    echo "  Session $ITER — $(date) — commit $COMMIT"
    echo "=============================================="

    claude --print "$(cat AGENT_PROMPT.md)" \
           --model claude-opus-4-6 \
           --dangerously-skip-permissions \
           > "$LOGFILE" 2>&1 || true

    echo "  Exit code: $?"
    echo "  Log: $LOGFILE ($(wc -c < "$LOGFILE") bytes)"

    # Print summary (last 20 lines of claude's output)
    echo "  --- Output tail ---"
    tail -20 "$LOGFILE"
    echo "  ---"

    # Safety commit+push
    if ! git diff --quiet || ! git diff --cached --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]; then
        echo "  Auto-committing remaining changes..."
        git add -A
        git commit -m "Auto-commit after agent session $ITER (${COMMIT})" 2>/dev/null || true
        git push 2>/dev/null || true
    else
        echo "  No uncommitted changes."
    fi

    echo ""
    sleep 5
done
