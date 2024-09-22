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
  --batch_size 16 \
  --master_batch 8 \
  --lr 5e-4 \
  --lr_step 90,120 \
  --num_epochs 140 \
  --val_intervals 1 \
  --gpus 0 \
  --num_workers 8 \
  --load_model ../models/ddd_3dop.pth
