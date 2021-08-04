#!/bin/bash

PYTHONUNBUFFERED=1
export PYTHONUNBUFFERED
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin
cp ../mjkey.txt /root/.mujoco/

bash scripts/train_nat.sh $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12}
