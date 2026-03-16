## Visualizzatore di frattali in Python

Visualizzatore interattivo con supporto **GPU (OpenCL)** per AMD Radeon.

### Frattali disponibili

- **Mandelbrot** (m)
- **Julia** (j)
- **Burning Ship** (b)
- **Tricorn** (t)
- **Newton** (n)

### Requisiti

```bash
pip install -r requirements.txt
```

### Esecuzione Python

```bash
# GPU
python fractal_viewer.py --backend opencl

# Fullscreen
python fractal_viewer.py --backend opencl --fullscreen

# Tipo iniziale
python fractal_viewer.py --type burning_ship --backend opencl
```

### 🚀 **Standalone .EXE (Windows - No Python needed!)**

**Pronto all'uso** `dist/fractal_viewer.exe` (~300MB, self-contained).

```cmd
dist\fractal_viewer.exe                           # Avvia Mandelbrot
dist\fractal_viewer.exe --backend opencl          # GPU se disponibile
dist\fractal_viewer.exe --type julia --fullscreen # Julia fullscreen
dist\fractal_viewer.exe --help                    # Aiuto
```

**Ricostruisci**:
```cmd
build_exe.bat
```

**Compatibile**: Windows 11+, copia ovunque.

### Controlli

| Tasto | Azione |
|-------|--------|
| **m** **j** **b** **t** **n** | Cambia frattale |
| **+** **-** | Zoom |
| **Frecce** | Sposta |
| **Click** | Centra |
| **r** | Reset |
| **[** **]** | Iterazioni |
| **c** | Colori |
| **s** | Salva PNG |
| **S** | Salva 4K |
| **a** | Anim GIF |
| **x** | Copia parametri |

**Julia**: 1-4 Re/Im(c), 5-9 presets.

### Colormap

inferno, viridis, plasma, etc. (c per ciclare).

**Repo commits**: 3 aggiunti per exe/build.

Enjoy!
