#!/bin/bash -e

cd ./src
python ./demo.py ctdet --demo ../images --load_model ../models/ctdet_coco_dla_2x.pth

