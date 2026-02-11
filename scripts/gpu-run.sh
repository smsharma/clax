#!/bin/bash
# gpu-run.sh — Run a command on the GPU compute node from the login node.
# Usage: bash scripts/gpu-run.sh "python scripts/gpu_planck_test.py"
#        bash scripts/gpu-run.sh "pytest tests/ --fast -x -q"
#
# Finds the running GPU batch job automatically and uses srun --overlap.

JOBID=$(squeue -u $USER -n gpu-backend -h -o "%i" | head -1)
if [ -z "$JOBID" ]; then
    echo "ERROR: No gpu-backend job found. Submit one with:"
    echo "  sbatch --partition=GPU-shared --gres=gpu:v100-32:1 --ntasks=1 --cpus-per-task=5 --mem=40G --time=48:00:00 --account=phy230064p --job-name=gpu-backend --wrap='sleep 172800'"
    exit 1
fi

NODE=$(squeue -j $JOBID -h -o "%N")
echo ">>> Running on GPU node $NODE (job $JOBID)" >&2

# Source conda env and run the command on the compute node
srun --jobid=$JOBID --overlap --pty bash -c "
    source /opt/packages/anaconda3-2022.10/etc/profile.d/conda.sh
    conda activate /ocean/projects/phy230064p/smishrasharma/.conda/envs/jaxclass 2>/dev/null
    export JAX_COMPILATION_CACHE_DIR=\${LOCAL:-/tmp}/jax_cache
    mkdir -p \$JAX_COMPILATION_CACHE_DIR
    cd /ocean/projects/phy230064p/smishrasharma/jaxclass
    $*
"
