#!/bin/bash
source /.env/bin/activate
python /home/vessel_detection/train.py "$@"