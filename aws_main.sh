#!/bin/bash
# make sure you modified the runner file accordingly
for i in {1..10}
do
python -W ignore runner.py
done

# make sure to enter the instance id here
aws ec2 stop-instances --instance-ids i-04efe05390440073f
