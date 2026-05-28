@echo off
call venv\Scripts\activate
python src\evaluate.py --dataset NEU-DET --draw
pause
