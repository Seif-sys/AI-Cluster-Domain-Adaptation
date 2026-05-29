#!/bin/bash
#SBATCH --job-name=coral_da
#SBATCH --partition=iai
#SBATCH --account=student
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --tasks=1
#SBATCH --threads=1
#SBATCH --gres=gpu:1
#SBATCH --verbose
#SBATCH --exclude=login
#SBATCH --output=coral_%j.out
#SBATCH --mail-user=<yourEmailId>@tu-braunschweig.de
#SBATCH --mail-type=INVALID_DEPEND,BEGIN,END,FAIL,TIME_LIMIT_50,TIME_LIMIT

# =============================================================================
# run_coral.sh
# Unsupervised Domain Adaptation with Deep CORAL on Colored-MNIST.
#
# Prerequisite: run_source.sh must have completed and produced:
#   ${WORK}/checkpoints/source/best_model.pt
#
# Usage (interactive, no SLURM):
#   bash scripts/run_coral.sh
#
# Usage (cluster):
#   sbatch scripts/run_coral.sh
# =============================================================================

# ---- Paths ------------------------------------------------------------------
BASE=/home/AppTainerImages
export BASE

WORK=/home/`whoami`/AppT
export WORK

VENV=${WORK}/myVenv
export VENV

PROJECT=${WORK}/domain-adaptation          # root of your git repo
export PROJECT

SOURCE_CKPT=${WORK}/checkpoints/source/best_model.pt
RESULTS_DIR=${WORK}/results/coral
CHECKPOINTS_DIR=${WORK}/checkpoints/coral

# ---- Sanity check: source checkpoint must exist ----------------------------
if [ ! -f "$SOURCE_CKPT" ]; then
    echo "ERROR: source checkpoint not found at $SOURCE_CKPT"
    echo "       Run run_source.sh first and wait for it to finish."
    exit 1
fi

# ---- Create working directory if missing ------------------------------------
[ ! -d $WORK ] && mkdir -p $WORK
[ ! -d $RESULTS_DIR ]     && mkdir -p $RESULTS_DIR
[ ! -d $CHECKPOINTS_DIR ] && mkdir -p $CHECKPOINTS_DIR

# ---- Create writable overlay if missing -------------------------------------
if [ ! -f ${WORK}/ubuntu_overlay12.img ]; then
    echo "Creating writable overlay..."
    cd $WORK
    apptainer overlay create --size 1024 --create-dir ${WORK} ${WORK}/ubuntu_overlay12.img
    echo "Overlay created."
fi

# ---- Install requirements if not already done --------------------------------
if [ ! -f $VENV/checked.log ]; then
    echo "Installing requirements into venv..."
    cp $BASE/check.py $BASE/checktorch.py $WORK
    apptainer shell --nv --overlay ${WORK}/ubuntu_overlay12.img ${BASE}/ubuntu-cuda12.sif <<ENDE
      python3 -m venv $VENV
      source ${VENV}/bin/activate
      pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
      pip install numpy scikit-learn matplotlib tqdm
      cd $VENV
      mv $WORK/check.py $WORK/checktorch.py .
      python3 check.py | tee checked.log
      python3 checktorch.py | tee -a checked.log
      rm check.py checktorch.py
ENDE
    echo "Requirements installed. See $VENV/checked.log"
fi

# ---- Run CORAL adaptation ---------------------------------------------------
echo "Starting Deep CORAL adaptation..."
echo "  Source checkpoint : $SOURCE_CKPT"
echo "  Results dir       : $RESULTS_DIR"
  echo "  Checkpoints dir   : $CHECKPOINTS_DIR"
echo "  Job ID            : $SLURM_JOB_ID"

apptainer exec --nv \
    --overlay ${WORK}/ubuntu_overlay12.img \
    ${BASE}/ubuntu-cuda12.sif \
    bash -c "
        source ${VENV}/bin/activate
        cd ${PROJECT}
        python src/train_coral.py \
            --source_ckpt     ${SOURCE_CKPT} \
            --backbone        cnn \
            --epochs          20 \
            --batch_size      64 \
            --lr              1e-3 \
            --lambda_coral    1.0 \
            --feat_dim        256 \
            --dropout         0.3 \
            --source_color_prob 0.99 \
            --target_color_prob 0.10 \
            --seed            42 \
            --num_workers     4 \
            --data_root       ${WORK}/data \
            --results_dir     ${RESULTS_DIR} \
            --checkpoints_dir ${CHECKPOINTS_DIR}
    "

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "CORAL adaptation finished successfully."
    echo "Results    : ${RESULTS_DIR}/results_summary.txt"
    echo "Metrics    : ${RESULTS_DIR}/metrics.csv"
    echo "Checkpoint : ${CHECKPOINTS_DIR}/best_model.pt"
else
    echo "ERROR: train_coral.py exited with code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "Job $SLURM_JOB_ID completed."