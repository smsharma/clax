#!/bin/bash
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=48:00:00
#SBATCH --account=phy230064p
#SBATCH --job-name=claude-tmux
#SBATCH --output=/ocean/projects/phy230064p/smishrasharma/jaxclass/loop_logs/claude_tmux_%j.out

# Setup env
source /opt/packages/anaconda3-2022.10/etc/profile.d/conda.sh
conda activate /ocean/projects/phy230064p/smishrasharma/.conda/envs/jaxclass
export PATH="$HOME/.local/bin:$PATH"
source "$HOME/.env"
export CI=true
export JAX_COMPILATION_CACHE_DIR=${LOCAL:-/tmp}/jax_cache
mkdir -p $JAX_COMPILATION_CACHE_DIR

cd /ocean/projects/phy230064p/smishrasharma/jaxclass

# Start tmux and claude inside it, then wait
tmux new-session -d -s claude "claude --dangerously-skip-permissions"

# Keep job alive while tmux exists
while tmux has-session -t claude 2>/dev/null; do
    sleep 60
done
