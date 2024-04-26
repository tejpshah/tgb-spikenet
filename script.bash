#!/bin/bash -l
#SBATCH --output=logfile
#SBATCH --mem=200g

cd /common/home/ac1771/Desktop/bic-project
conda activate dhypr-lib
python3 spikenet_main.py --dataset tgbn --tgbn_dataset tgbn-genre --hids 128 10 --batch_size 16 --p 0.5 --train_size 0.4