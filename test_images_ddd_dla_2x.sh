#!/bin/bash -e

work_dir=$(
  cd $(dirname $0)
  pwd
)
cd $work_dir
echo "WorkDir: $work_dir"

source ./activate_env.sh pytorch

python3 ./src/demo.py ddd \
  --arch dla_34 \
  --demo ../images \
  --load_model ../models/ddd_3dop.pth # 按空格
