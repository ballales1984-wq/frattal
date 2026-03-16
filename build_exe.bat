@echo off
rem Build standalone fractal_viewer.exe
rem Using Python 3.11 for better compatibility with matplotlib 3.6.3
set PYTHON=C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe

if exist fractal_exe_env rmdir /s /q fractal_exe_env
%PYTHON% -m venv fractal_exe_env
fractal_exe_env\Scripts\pip install --upgrade pip
fractal_exe_env\Scripts\pip install -r requirements_exe.txt
fractal_exe_env\Scripts\pyinstaller --onefile --windowed --clean --name fractal_viewer fractal_viewer.py
echo.
echo Build complete: dist\fractal_viewer.exe ^(self-contained, no Python needed^)
echo Test: dist\fractal_viewer.exe
echo.
pause
