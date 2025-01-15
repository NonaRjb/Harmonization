#!/usr/bin/env bash
#SBATCH -A berzelius-2024-324
#SBATCH --mem 50GB
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -t 2-00:00:00
#SBATCH --mail-type FAIL
#SBATCH --mail-user nonar@kth.se
#SBATCH --output /proj/rep-learning-robotics/users/x_nonra/Harmonization/logs/%J_slurm.out
#SBATCH --error  /proj/rep-learning-robotics/users/x_nonra/Harmonization/logs/%J_slurm.err

data_path="/proj/rep-learning-robotics/users/x_nonra/alignvis/data/things_eeg_2/images"
save_path="/proj/rep-learning-robotics/users/x_nonra/alignvis/data/things_eeg_2/image_embeddings"
model_name="levit"

cd /proj/rep-learning-robotics/users/x_nonra/Harmonization

# Check job environment
echo "JOB:  ${SLURM_JOB_ID}"
echo "TASK: ${SLURM_ARRAY_TASK_ID}"
echo "HOST: $(hostname)"
echo ""

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate harmonization

python gen_harmonized_embeddings.py --data_root "$data_path" --embeddings_dir "$save_path" --model_name "$model_name" --human_aligned