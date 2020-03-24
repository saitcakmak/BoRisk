#!/bin/bash
# make sure you modified the runner file accordingly
python -W ignore runner.py

# make sure to enter the instance id here
aws ec2 stop-instances --instance-ids i-05d31175b5c0bc5d2
