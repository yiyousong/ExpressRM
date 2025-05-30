#!/bin/bash -l
#SBATCH --qos 6gpus
#SBATCH --partition gpu3090
#SBATCH --ntasks=6
#SBATCH --cpus-per-task 4
#SBATCH --time=30-00:00:00
#SBATCH --gpus-per-task 1
#SBATCH --mem-per-gpu=10G
#SBATCH --mail-type=All
#SBATCH --mail-user yiyou.song15@student.xjtlu.edu.cn
#SBATCH --job-name=ExpressRM
#SBATCH --output=out.out
module load anaconda3
source activate /gpfs/work/bio/jiameng/.conda/env/torch-env
srun -N 1 -n 1 -G 1 -c 4 python AdaptRM_ExpressRMversion.py &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,1,1,1,1 --epoch 1000 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,0,0 --testlist 0,1,2,3 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,1,0 --testlist 0,1,2,3 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,1,0,0,0 --testlist 0,1,2,3 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,1,0,0 --testlist 0,1,2,3 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,0,1 --testlist 0,1,2,3 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,1,0,1 --testlist 0,1,2,3 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,1,1,1,1 --testlist 0,1,2,3 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,0,0 --testlist 4,5,6,7 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,1,0 --testlist 4,5,6,7 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,1,0,0,0 --testlist 4,5,6,7 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,1,0,0 --testlist 4,5,6,7 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,0,1 --testlist 4,5,6,7 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,1,0,1 --testlist 4,5,6,7 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,1,1,1,1 --testlist 4,5,6,7 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,0,0 --testlist 8,9,10,11 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,1,0 --testlist 8,9,10,11 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,1,0,0,0 --testlist 8,9,10,11 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,1,0,0 --testlist 8,9,10,11 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,0,1 --testlist 8,9,10,11 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,1,0,1 --testlist 8,9,10,11 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,1,1,1,1 --testlist 8,9,10,11 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,0,0 --testlist 12,13,14,15 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,1,0 --testlist 12,13,14,15 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,1,0,0,0 --testlist 12,13,14,15 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,1,0,0 --testlist 12,13,14,15 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,0,1 --testlist 12,13,14,15 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,1,0,1 --testlist 12,13,14,15 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,1,1,1,1 --testlist 12,13,14,15 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,0,0 --testlist 16,17,18,19 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,1,0 --testlist 16,17,18,19 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,1,0,0,0 --testlist 16,17,18,19 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,1,0,0 --testlist 16,17,18,19 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,0,1 --testlist 16,17,18,19 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,1,0,1 --testlist 16,17,18,19 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,1,1,1,1 --testlist 16,17,18,19 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,0,0 --testlist 20,21,22,23 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,1,0 --testlist 20,21,22,23 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,1,0,0,0 --testlist 20,21,22,23 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,1,0,0 --testlist 20,21,22,23 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,0,1 --testlist 20,21,22,23 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,1,0,1 --testlist 20,21,22,23 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,1,1,1,1 --testlist 20,21,22,23 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,0,0 --testlist 24,25,26,27 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,1,0 --testlist 24,25,26,27 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,1,0,0,0 --testlist 24,25,26,27 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,1,0,0 --testlist 24,25,26,27 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,0,1 --testlist 24,25,26,27 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,1,0,1 --testlist 24,25,26,27 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,1,1,1,1 --testlist 24,25,26,27 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,0,0 --testlist 28,29,30,31 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,1,0 --testlist 28,29,30,31 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,1,0,0,0 --testlist 28,29,30,31 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,1,0,0 --testlist 28,29,30,31 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,0,1 --testlist 28,29,30,31 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,1,0,1 --testlist 28,29,30,31 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,1,1,1,1 --testlist 28,29,30,31 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,0,0 --testlist 32,33,34,35,36 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,1,0 --testlist 32,33,34,35,36 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,1,0,0,0 --testlist 32,33,34,35,36 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,1,0,0 --testlist 32,33,34,35,36 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,0,1 --testlist 32,33,34,35,36 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,1,0,1 --testlist 32,33,34,35,36 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,1,1,1,1 --testlist 32,33,34,35,36 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,1,1  --testlist 0,1,2,3 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,1,1  --testlist 4,5,6,7 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,1,1  --testlist 8,9,10,11 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,1,1  --testlist 12,13,14,15 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,1,1  --testlist 16,17,18,19 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,1,1  --testlist 20,21,22,23 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,1,1  --testlist 24,25,26,27 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,1,1  --testlist 28,29,30,31 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,0,1,1  --testlist 32,33,34,35,36 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,1,1,1  --testlist 0,1,2,3 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,1,1,1  --testlist 4,5,6,7 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,1,1,1  --testlist 8,9,10,11 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,1,1,1  --testlist 12,13,14,15 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,1,1,1  --testlist 16,17,18,19 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,1,1,1  --testlist 20,21,22,23 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,1,1,1  --testlist 24,25,26,27 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,1,1,1  --testlist 28,29,30,31 &
srun -N 1 -n 1 -G 1 -c 4 python ExpressRM.py --featurelist 1,0,1,1,1  --testlist 32,33,34,35,36 &
wait