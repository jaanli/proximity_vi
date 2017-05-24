#!/bin/bash
JOB_CMD="./sigmoid_belief_network_experiment.sh"
## With slurm scheduler:
# JOB_CMD="sbatch launch_job_$HOSTNAME.cmd $JOB_CMD"
echo $JOB_CMD

EXPERIMENT=sigmoid_belief_network_grid
SLEEP=0.1

for N_LAYERS in 1 3
do
  for W_EPS in -100 0
  do
    if [ $W_EPS -eq -100 ]
    then
      PRIOR=0.001
    elif [ $W_EPS -eq 0 ]
    then
      PRIOR=0.5
    fi
    $JOB_CMD --inference vanilla \
       --p/w_eps $W_EPS \
       --p/n_layers $N_LAYERS \
       --p/bernoulli_p $PRIOR \
       --log/experiment $EXPERIMENT
    sleep $SLEEP
    for DECAY_RATE in 1e-5 1e-6 1e-7 1e-8 1e-9 1e-10 1e-20 1e-30
    do
      for STATISTIC in entropy mean_variance kl
        do
        $JOB_CMD --inference proximity \
            --p/n_layers $N_LAYERS \
            --p/w_eps $W_EPS \
            --c/proximity_statistic $STATISTIC \
            --c/lag moving_average \
            --c/decay_rate $DECAY_RATE \
            --log/experiment $EXPERIMENT \
            --p/bernoulli_p $PRIOR \
            --moving_average/decay 0.9999
        sleep $SLEEP
	$JOB_CMD --inference vanilla \
	    --p/n_layers $N_LAYERS \
	    --p/w_eps $W_EPS \
	    --c/decay_rate $DECAY_RATE \
	    --log/experiment $EXPERIMENT \
	    --c/decay exponential \
	    --optim/deterministic_annealing True \
	    --p/bernoulli_p $PRIOR
	sleep $SLEEP
      done
    done
  done
done
