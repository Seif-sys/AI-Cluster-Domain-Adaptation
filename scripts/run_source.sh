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
#SBATCH --mail-user=s.saad@tu-braunschweig.de
#SBATCH --mail-type=INVALID_DEPEND,BEGIN,END,FAIL,TIME_LIMIT_50,TIME_LIMIT

# =============================================================================
# Use local scratch space instead of home directory
# =============================================================================

# Use SLURM temporary directory (local SSD on compute node)
export WORK=$SLURM_TMPDIR
export PROJECT_DIR=/home/y0108835/GITZ-home/AppData/AI-Cluster-Domain-Adaptation

echo "Using WORK directory: $WORK"
echo "Project directory: $PROJECT_DIR"

# Create working directories in local scratch
mkdir -p $WORK/results/source
mkdir -p $WORK/checkpoints/source
mkdir -p $WORK/data
mkdir -p $WORK/myVenv

# Check if Apptainer is available
module load apptainer 2>/dev/null || echo "Apptainer module not found"

# Check for container
CONTAINER=/home/AppTainerImages/ubuntu-cuda12.sif
if [ ! -f $CONTAINER ]; then
    echo "Container not found. Please check path."
    ls -la /home/AppTainerImages/ 2>/dev/null || echo "Directory not accessible"
    exit 1
fi

# Copy necessary files to local scratch
cp -r $PROJECT_DIR/src $WORK/
cp -r $PROJECT_DIR/scripts $WORK/
cp $PROJECT_DIR/*.py $WORK/ 2>/dev/null

# Create fresh overlay image in local scratch (writable!)
cd $WORK
apptainer overlay create --size 2048 ubuntu_overlay.img

# Install requirements in container
apptainer exec --nv --overlay $WORK/ubuntu_overlay.img $CONTAINER << 'EOF'
    python3 -m venv $SLURM_TMPDIR/myVenv
    source $SLURM_TMPDIR/myVenv/bin/activate
    pip install --upgrade pip
    pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
    pip install numpy scikit-learn matplotlib tqdm
EOF

# Run training
apptainer exec --nv --overlay $WORK/ubuntu_overlay.img $CONTAINER \
    bash -c "
        source $WORK/myVenv/bin/activate
        cd $WORK
        python src/train_source.py \
            --backbone cnn \
            --epochs 20 \
            --batch_size 64 \
            --lr 1e-3 \
            --feat_dim 256 \
            --dropout 0.3 \
            --source_color_prob 0.99 \
            --target_color_prob 0.10 \
            --seed 42 \
            --num_workers 4 \
            --data_root $WORK/data \
            --results_dir $WORK/results/source \
            --checkpoints_dir $WORK/checkpoints/source
    "

# Copy results back to home directory after training
cp -r $WORK/results/source /home/y0108835/GITZ-home/AppData/AI-Cluster-Domain-Adaptation/results/
cp -r $WORK/checkpoints/source /home/y0108835/GITZ-home/AppData/AI-Cluster-Domain-Adaptation/checkpoints/

echo "Training completed. Results copied back to home directory."