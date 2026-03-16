# TODO: Build Standalone Fractal Viewer .exe for Windows

Status: Deps installed (numpy 2.1.3, matplotlib 3.9.2, pyopencl/imageio/pyinstaller). PyInstaller build running (~10-30min for numpy/matplotlib bundling + OpenCL hooks). Progress: Analyzing modules.

Repo commits up-to-date (cc77f70).

- [x] Step 1: Create TODO.md 
- [x] Step 2: Commit current modifications (.history/, README.md, fractal_viewer.py)
- [x] Step 3: Create/add to .gitignore for venv/build/dist
- [x] Step 4: Create isolated venv (python -m venv fractal_exe_env)
- [x] Step 5: Activate venv & install deps (pip install -r requirements.txt pyinstaller)
- [ ] Step 6: Build exe (pyinstaller --onefile --windowed --name fractal_viewer fractal_viewer.py) [RUNNING]
- [ ] Step 7: Test dist/fractal_viewer.exe (run GUI, check OpenCL/CPU fallback)
- [ ] Step 8: Update README.md with build/run exe instructions + build.bat script
- [ ] Step 9: Commit/push build.bat + README update, complete task

