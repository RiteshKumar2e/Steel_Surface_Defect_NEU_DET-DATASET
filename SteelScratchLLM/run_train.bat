@echo off
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
python src\train.py --dataset NEU-DET --epochs 35
pause
