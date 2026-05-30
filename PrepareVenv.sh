#!/bin/bash
#SBATCH --job-name=meinTollerJobName
#SBATCH --partition=iai
#SBATCH --account=student
#SBATCH --time=00:10:00
#SBATCH --nodes=1  
#SBATCH --cpus-per-task=4  # don't use too much as it gets more expensive...
#SBATCH --mem=12G           # the same here
#SBATCH --tasks=1  
#SBATCH --threads=1   
#SBATCH --gres=shard:1    # einen Teil einer irgendeiner GPU
#SBATCH --verbose
#SBATCH --exclude=login   # don't try to run anything on login node!!
#SBATCH --output=prepare_venv_%j.out
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


# WHERE Is the Base Image? There it is..
BASE=/home/AppTainerImages
export BASE

# where will my work directory be? 
WORK=/home/`whoami`/AppT    # or /home/`whoami`/GITZ-home/somewhere
export WORK

# where will my Venv be located?
VENV=${WORK}/myVenv
export VENV

# if missing create working directory
[ ! -d $WORK ] && mkdir -p $WORK

# create writable overlay in working directory
if [ ! -f ${WORK}/ubuntu_overlay12.img ]; then
    echo "build writable overlay with personally needed python modules"
    cd $WORK
    apptainer overlay create --size 1024 --create-dir ${WORK} ${WORK}/ubuntu_overlay12.img
    echo "build overlay done"
fi

# install requirements and check
# check what you need, this is an example only
# (!) use the below mentioned torch version, to be able to run on any of our GPUs
if [ ! -f $VENV/checked.log ]; then
    echo "install requirements in venv"
    # get my demo python here
    cp $BASE/check.py $BASE/checktorch.py $WORK    ## (!) VENV might not exist yet 
    apptainer shell --nv --overlay ${WORK}/ubuntu_overlay12.img ${BASE}/ubuntu-cuda12.sif <<ENDE
    python3 -m venv $VENV
    source ${VENV}/bin/activate
    pip install cupy
    pip install numba
    pip install numpy
    pip install matplotlib scikit-learn tqdm pandas
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
    cd $VENV
    mv $WORK/check.py $WORK/checktorch.py .
    python3 check.py | tee checked.log          # if tests fail completely checked.log might be empty; so if its empty     
    python3 checktorch.py | tee -a checked.log  # remove it manually to get here again. remove it if requirements changed.
            python3 - <<PY | tee -a checked.log
    import torch
    import torchvision
    import numpy as np

    print("Torch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)
    print("NumPy:", np.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    PY
    rm check.py checktorch.py
ENDE
    echo "installed requirements and did checks; see $VENV/checked.log"

fi


echo "Batch Job completed" 

