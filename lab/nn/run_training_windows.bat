@echo off
REM Run training.py with different splits on Windows in parallel
start python training.py --os windows --split 0.001 0.001 0.998 --model-path model_split_0001_0001_0998.pth
start python training.py --os windows --split 0.01 0.01 0.98 --model-path model_split_001_001_098.pth
start python training.py --os windows --split 0.1 0.1 0.8 --model-path model_split_01_01_08.pth
start python training.py --os windows --split 0.2 0.2 0.6 --model-path model_split_02_02_06.pth
start python training.py --os windows --split 0.4 0.4 0.2 --model-path model_split_04_04_02.pth
start python training.py --os windows --split 0.8 0.2 0.0 --model-path model_split_08_02_00.pth
