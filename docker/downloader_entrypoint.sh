#!/bin/bash
source /.env/bin/activate
python /home/vessel_detection/download_imagery.py "$@"