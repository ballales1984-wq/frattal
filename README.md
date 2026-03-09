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

### Esecuzione

```bash
# GPU
python fractal_viewer.py --backend opencl

# Fullscreen
python fractal_viewer.py --backend opencl --fullscreen

# Tipo iniziale
python fractal_viewer.py --type burning_ship --backend opencl
```

### Controlli

| Tasto | Azione |
|-------|--------|
| **m** **j** **b** **t** **n** | Mandelbrot, Julia, Burning Ship, Tricorn, Newton |
| **+** **-** | Zoom |
| **Frecce** | Spostamento |
| **Click** | Centra |
| **r** | Reset vista |
| **[** **]** | Iterazioni |
| **c** | Colormap |
| **s** | Salva |
| **S** | Salva 4K |
| **a** | Animazione zoom (GIF) |
| **u** | Indietro zoom |
| **x** | Copia parametri |
| **1-4** | Julia: Re/Im(c) |
| **5-9** | Julia: preset (dendrite, spirale, drago, elefante, snowflake) |

### Colormap

`inferno`, `viridis`, `plasma`, `magma`, `turbo`, `hot`, `cool`, `winter`, `spring`, `summer`, `autumn`, `twilight`, `Spectral`, `RdYlBu`
