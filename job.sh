#!/bin/bash
#SBATCH --output=./outputs/output_logs/ci_run5.out
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:0
#SBATCH --partition=long-cpu
#SBATCH --begin=now

source $HOME/.r1_hackathon/bin/activate
cd $HOME/r1_hackathon
python get_comparative_illusion_results.py -run_name "run5" -input_directory "CIdataset.csv" -out_dir "outputs/comparative_illusion"