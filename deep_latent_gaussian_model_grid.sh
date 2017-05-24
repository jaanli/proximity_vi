#!/bin/bash
JOB_CMD="./deep_latent_gaussian_model_experiment.sh"
## With slurm scheduler:
# JOB_CMD="sbatch launch_job_$HOSTNAME.cmd $JOB_CMD"
echo "Using $JOB_CMD"

EXPERIMENT=deep_latent_gaussian_model_grid

$JOB_CMD --inference vanilla --log/experiment $EXPERIMENT
sleep 1

for MAGNITUDE in 1 1e-1 1e-2 1e-3 1e-4 1e-5
do
  $JOB_CMD --inference proximity \
      --c/proximity_statistic orthogonal \
      --c/magnitude $MAGNITUDE \
      --log/experiment $EXPERIMENT
  sleep 1
done
