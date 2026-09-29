@echo off
rem 매일 앞으로 기록 (Windows 작업 스케줄러에서 매일 09:00 KST 실행)
rem SPX0DTE_HOME = 이 저장소 폴더, PYTHON = databento가 설치된 파이썬
cd /d %SPX0DTE_HOME%
%PYTHON% -W ignore scripts\40_forward_log.py --go --budget 0.05 >> output\forward_run.log 2>&1
%PYTHON% -W ignore scripts\41_forward_charts.py >> output\forward_run.log 2>&1
