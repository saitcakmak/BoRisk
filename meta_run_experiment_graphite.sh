#!/bin/sh

for i in $(seq 12 30)
do 
   sleep 1
   sbatch --requeue /home/ra598/Raul/Projects/BoRisk/run_experiment_graphite.sub $i
done
