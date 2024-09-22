#!/bin/bash -e

work_dir=$(cd `dirname $0`; pwd)
cd $work_dir
echo "WorkDir: $work_dir"

source ./activate_env.sh pytorch

cd ./src
python3 ./demo.py ctdet --demo ../images --load_model ../models/ctdet_coco_dla_2x.pth

