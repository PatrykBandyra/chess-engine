#!/bin/bash
# Run training.py with different splits on Linux in parallel
python3 training.py --os linux --split 0.001 0.001 0.998 --model-path model_split_0001_0001_0998.pth &
python3 training.py --os linux --split 0.01 0.01 0.98 --model-path model_split_001_001_098.pth &
python3 training.py --os linux --split 0.1 0.1 0.8 --model-path model_split_01_01_08.pth &
wait
