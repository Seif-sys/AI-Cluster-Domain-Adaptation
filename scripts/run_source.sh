#!/bin/bash
#SBATCH --job-name=source_da
#SBATCH --partition=iai
#SBATCH --account=student
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --tasks=1
#SBATCH --threads=1
#SBATCH --gres=gpu:1
#SBATCH --verbose
#SBATCH --exclude=login
#SBATCH --output=source_%j.out
#SBATCH --mail-user=<yourEmailId>@tu-braunschweig.de
#SBATCH --mail-type=INVALID_DEPEND,BEGIN,END,FAIL,TIME_LIMIT_50,TIME_LIMIT

# =============================================================================
# run_source.sh
# Train the source-only CNN baseline on Colored-MNIST (binary classification).
#
# This is the first script to run. It produces:
#   ${WORK}/checkpoints/source/best_model.pt  <- required by run_coral.sh
#                                                and run_upper_bound.sh
#   ${WORK}/results/source/results_summary.txt
#   ${WORK}/results/source/metrics.csv
#
# Usage (interactive, no SLURM):
#   bash scripts/run_source.sh
#
# Usage (cluster):
#   sbatch scripts/run_source.sh
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

RESULTS_DIR=${WORK}/results/source
CHECKPOINTS_DIR=${WORK}/checkpoints/source

# ---- Create working directory if missing ------------------------------------
[ ! -d $WORK ]            && mkdir -p $WORK
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

# ---- Run source-only training -----------------------------------------------
echo "Starting source-only baseline training..."
echo "  Results dir     : $RESULTS_DIR"
echo "  Checkpoints dir : $CHECKPOINTS_DIR"
echo "  Job ID          : $SLURM_JOB_ID"

apptainer exec --nv \
    --overlay ${WORK}/ubuntu_overlay12.img \
    ${BASE}/ubuntu-cuda12.sif \
    bash -c "
        source ${VENV}/bin/activate
        cd ${PROJECT}
        python src/train_source.py \
            --backbone          cnn \
            --epochs            20 \
            --batch_size        64 \
            --lr                1e-3 \
            --feat_dim          256 \
            --dropout           0.3 \
            --source_color_prob 0.99 \
            --target_color_prob 0.10 \
            --seed              42 \
            --num_workers       4 \
            --data_root         ${WORK}/data \
            --results_dir       ${RESULTS_DIR} \
            --checkpoints_dir   ${CHECKPOINTS_DIR}
    "

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Source training finished successfully."
    echo "Results    : ${RESULTS_DIR}/results_summary.txt"
    echo "Metrics    : ${RESULTS_DIR}/metrics.csv"
    echo "Checkpoint : ${CHECKPOINTS_DIR}/best_model.pt"
    echo ""
    echo "Next step  : sbatch scripts/run_coral.sh"
    echo "             sbatch scripts/run_upper_bound.sh"
else
    echo "ERROR: train_source.py exited with code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo "Job $SLURM_JOB_ID completed."