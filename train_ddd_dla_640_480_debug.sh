#!/bin/bash -e

work_dir=$(
  cd $(dirname $0)
  pwd
)
cd $work_dir
echo "WorkDir: $work_dir"

source ./activate_env.sh pytorch

python3 ./src/main.py ddd \
  --exp_id kitti_dla_2x \
  --arch dla_34 \
  --dataset kitti \
  --batch_size 4 \
  --master_batch 4 \
  --lr 5e-4 \
  --lr_step 90,120 \
  --num_epochs 1 \
  --num_iters 1 \
  --val_intervals 1 \
  --debug 4 \
  --gpus 0 \
  --num_workers 8 \
  --load_model ../models/ddd_3dop.pth
