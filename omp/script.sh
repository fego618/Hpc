#!/bin/bash
#SBATCH --job-name testJohn
#SBATCH --time 05:00
#SBATCH --nodes 1
#SBATCH --output johnOsorio.out

echo "The job has begun."
echo "Wait one minute..."
sleep 30
echo "Wait a second minute..."
sleep 30
echo "Wait a third minute..."
sleep 30
echo "Enough waiting: job completed."
