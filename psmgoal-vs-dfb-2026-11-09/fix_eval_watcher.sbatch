#!/bin/bash
#SBATCH --job-name=fixwatch
#SBATCH --output=/mnt/home/amohan/psm-data/psmgoal-vs-dfb-2026-11-09/slurm/%x-%j.out
#SBATCH --error=/mnt/home/amohan/psm-data/psmgoal-vs-dfb-2026-11-09/slurm/%x-%j.err
#SBATCH --partition=kisski-inference
#SBATCH --account=general
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
bash /mnt/home/amohan/git/Austin/PSMFLows/psmgoal-vs-dfb-2026-11-09/fix_eval_watcher.sh
