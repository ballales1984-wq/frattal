@echo off
rem Build standalone fractal_viewer.exe
if exist fractal_exe_env rmdir /s /q fractal_exe_env
python -m venv fractal_exe_env
call fractal_exe_env\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller
pyinstaller --onefile --windowed --clean --name fractal_viewer fractal_viewer.py
echo.
echo Build complete: dist\fractal_viewer.exe ^(self-contained, no Python needed^)
echo Test: dist\fractal_viewer.exe
echo.
pause

