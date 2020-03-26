#!/bin/bash
# make sure you modified the runner file accordingly
for i in {1..10}
do
python -W ignore runner.py
done

# make sure to enter the instance id here
aws ec2 stop-instances --instance-ids i-0aa62c4f65b5780b2
