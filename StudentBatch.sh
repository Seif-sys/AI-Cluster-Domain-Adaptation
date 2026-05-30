#!/bin/bash
#SBATCH --job-name=meinTollerJobName
#SBATCH --partition=iai
#SBATCH --account=student
#SBATCH --nodes=1  
#SBATCH --cpus-per-task=2  # don't use too much as it gets more expensive...
#SBATCH --mem=4G           # the same here
#SBATCH --tasks=1  
#SBATCH --threads=1  
#SBATCH --time=00:20:00   # Für Tests später ohne diese Angabe, dann default=10 Min. 
#SBATCH --gres=shard:1    # einen Teil einer irgendeiner GPU
#SBATCH --verbose
#SBATCH --exclude=login   # don't try to run anything on login node!!
#SBATCH --output=training_%j.out
#SBATCH --mail-user=m.labidi@tu-braunschweig.de
#SBATCH --mail-type=INVALID_DEPEND,BEGIN,END,FAIL,TIME_LIMIT_50,TIME_LIMIT
#----------
#-- --mail-type: any useful combination of those: 
#--       NONE, BEGIN, END, FAIL, REQUEUE, ALL (equivalent to BEGIN, END, FAIL, 
#--       INVALID_DEPEND, REQUEUE, and STAGE_OUT), INVALID_DEPEND (dependency 
#--       never satisfied), STAGE_OUT (burst buffer stage out and teardown 
#--       completed), TIME_LIMIT, TIME_LIMIT_90 (reached 90 percent of time 
#--       limit), TIME_LIMIT_80 (reached 80 percent of time limit), TIME_LIMIT_50
#-- --gres=gpu:1 : grabs a complete GPU which mioght be expensive 
#----------

echo "Hello from $(hostname) at $(date)"
TRAIN_SCRIPT=${1:-src/train_source.py}

export WORK=$HOME/AppT
export BASE=/home/AppTainerImages
export REPO=/home/y0113643/AI-Cluster-Domain-Adaptation

echo "start application"
[ ! -f ${WORK}/ubuntu_overlay12.img ] && { echo "*** please create overlay first. Run PrepareVenv.sh. ***"; exit 1; }

# (!) if you want to use (parts) of a gpu don't forget the --nv flag!

apptainer shell --nv --overlay ${WORK}/ubuntu_overlay12.img ${BASE}/ubuntu-cuda12.sif <<ENDE
  source ${WORK}/myVenv/bin/activate

  cd ${REPO}

  echo "Current directory:"
  pwd

  echo "Using GPU:"
  nvidia-smi -L

  echo "Python:"
  which python

  echo "PyTorch CUDA check:"
  python - <<PY
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

  echo "Running script: ${TRAIN_SCRIPT}"
  python ${TRAIN_SCRIPT}
ENDE

echo "application done "

echo "Batch Job completed" 

