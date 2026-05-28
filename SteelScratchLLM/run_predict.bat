@echo off
call venv\Scripts\activate
python src\predict.py --image sample_data\test.jpg --dataset NEU-DET --save results\prediction.jpg
pause
