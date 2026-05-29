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
#SBATCH --output=meinJob_%j.out
#SBATCH --mail-user=<volle email>@tu-braunschweig.de
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

export WORK=$HOME/AppT
export BASE=/home/AppTainerImages

echo "start application"
[ ! -f ${WORK}/ubuntu_overlay.img ] && { echo "*** please create overlay first. ***; exit 1; }

# (!) if you want to use (parts) of a gpu don't forget the --nv flag!
apptainer shell --nv --overlay ${WORK}/ubuntu_overlay12.img ${BASE}/ubuntu-cuda12.sif <<ENDE
  echo "Using $(nvidia-smi -L)"
  cd AppT/myVenv
  source bin/activate    cd app
  python torchtest.py
ENDE

echo "application done "

echo "Batch Job completed" 

