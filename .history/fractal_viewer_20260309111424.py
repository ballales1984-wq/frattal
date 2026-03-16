"""Visualizzatore interattivo di frattali Mandelbrot/Julia con backend CPU/OpenCL."""

import argparse
import glob
import os
import re
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    import pyopencl as cl

    _HAS_OPENCL = True
except Exception:
    cl = None
    _HAS_OPENCL = False

# Preset Julia (nome: (Re(c), Im(c)))
JULIA_PRESETS = {
    "dendrite": (-0.5, 0.5),
    "spirale": (-0.8, 0.156),
    "drago": (0.285, 0.01),
    "elefante": (0.3, 0.0),
    "snowflake": (0.25, 0.0),
}


def _make_grid(
    width: int,
    height: int,
    x_center: float,
    y_center: float,
    zoom: float,
    base_scale: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Costruisce una griglia 2D nel piano complesso centrata in (x_center, y_center)."""
    aspect_ratio = width / height
    scale = base_scale / zoom
    x_min = x_center - scale * aspect_ratio
    x_max = x_center + scale * aspect_ratio
    y_min = y_center - scale
    y_max = y_center + scale

    x_vals = np.linspace(x_min, x_max, width)
    y_vals = np.linspace(y_min, y_max, height)
    return np.meshgrid(x_vals, y_vals)


def generate_mandelbrot_cpu(
    width: int = 800,
    height: int = 600,
    max_iter: int = 100,
    x_center: float = -0.5,
    y_center: float = 0.0,
    zoom: float = 1.0,
) -> np.ndarray:
    """Calcola il set di Mandelbrot su CPU con NumPy."""
    x_vals, y_vals = _make_grid(width, height, x_center, y_center, zoom)
    c_vals = x_vals + 1j * y_vals

    z_vals = np.zeros_like(c_vals, dtype=complex)
    iters = np.zeros(c_vals.shape, dtype=int)

    mask = np.ones(c_vals.shape, dtype=bool)
    for i in range(max_iter):
        z_vals[mask] = z_vals[mask] * z_vals[mask] + c_vals[mask]
        escaped = np.abs(z_vals) > 2.0
        newly_escaped = escaped & mask
        iters[newly_escaped] = i
        mask &= ~escaped
        if not mask.any():
            break

    iters[mask] = max_iter
    return iters


def generate_mandelbrot_opencl(
    width: int = 800,
    height: int = 600,
    max_iter: int = 100,
    x_center: float = -0.5,
    y_center: float = 0.0,
    zoom: float = 1.0,
) -> np.ndarray:
    """Calcola il set di Mandelbrot su GPU tramite OpenCL, se disponibile."""
    if not _HAS_OPENCL:
        raise RuntimeError(
            "Backend OpenCL richiesto ma pyopencl non è disponibile. "
            "Installa pyopencl o usa --backend cpu."
        )

    aspect_ratio = width / height
    scale = 1.5 / zoom
    x_min = x_center - scale * aspect_ratio
    x_max = x_center + scale * aspect_ratio
    y_min = y_center - scale
    y_max = y_center + scale

    platforms = cl.get_platforms()
    devices = []
    for platform in platforms:
        try:
            devices.extend(platform.get_devices(device_type=cl.device_type.GPU))
        except Exception:
            continue
    if not devices:
        # fallback: qualsiasi device disponibile
        for platform in platforms:
            try:
                devices.extend(platform.get_devices())
            except Exception:
                continue
    if not devices:
        raise RuntimeError("Nessun dispositivo OpenCL disponibile sulla macchina.")

    # Usa il primo dispositivo trovato (tipicamente la GPU principale)
    selected_device = devices[0]
    print(f"Backend OpenCL attivo su dispositivo: {selected_device.name}")
    ctx = cl.Context(devices=[selected_device])
    queue = cl.CommandQueue(ctx)

    kernel_source = """
    __kernel void mandelbrot(
        const int width,
        const int height,
        const int max_iter,
        const float x_min,
        const float x_max,
        const float y_min,
        const float y_max,
        __global int* output)
    {
        int gx = get_global_id(0);
        int gy = get_global_id(1);
        if (gx >= width || gy >= height) {
            return;
        }

        float x0 = x_min + (x_max - x_min) * ((float)gx / (float)(width - 1));
        float y0 = y_min + (y_max - y_min) * ((float)gy / (float)(height - 1));

        float x = 0.0f;
        float y = 0.0f;
        int iter = 0;

        while (x * x + y * y <= 4.0f && iter < max_iter) {
            float xt = x * x - y * y + x0;
            y = 2.0f * x * y + y0;
            x = xt;
            iter++;
        }

        int idx = gy * width + gx;
        output[idx] = iter;
    }
    """

    program = cl.Program(ctx, kernel_source).build()

    result = np.empty(width * height, dtype=np.int32)
    buf_output = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, result.nbytes)

    kernel = program.mandelbrot
    kernel.set_args(
        np.int32(width),
        np.int32(height),
        np.int32(max_iter),
        np.float32(x_min),
        np.float32(x_max),
        np.float32(y_min),
        np.float32(y_max),
        buf_output,
    )

    global_size = (width, height)
    cl.enqueue_nd_range_kernel(queue, kernel, global_size, None)
    cl.enqueue_copy(queue, result, buf_output)
    queue.finish()

    return result.reshape((height, width))


def generate_julia_opencl(
    width: int = 800,
    height: int = 600,
    max_iter: int = 100,
    x_center: float = 0.0,
    y_center: float = 0.0,
    zoom: float = 1.0,
    c: complex = complex(-0.8, 0.156),
) -> np.ndarray:
    """Calcola Julia su GPU tramite OpenCL."""
    if not _HAS_OPENCL:
        raise RuntimeError("PyOpenCL non disponibile. Usa --backend cpu.")

    aspect_ratio = width / height
    scale = 1.5 / zoom
    x_min = x_center - scale * aspect_ratio
    x_max = x_center + scale * aspect_ratio
    y_min = y_center - scale
    y_max = y_center + scale
    cre, cim = float(c.real), float(c.imag)

    platforms = cl.get_platforms()
    devices = []
    for platform in platforms:
        try:
            devices.extend(platform.get_devices(device_type=cl.device_type.GPU))
        except Exception:
            continue
    if not devices:
        for platform in platforms:
            try:
                devices.extend(platform.get_devices())
            except Exception:
                continue
    if not devices:
        raise RuntimeError("Nessun dispositivo OpenCL disponibile.")
    ctx = cl.Context(devices=[devices[0]])
    queue = cl.CommandQueue(ctx)

    kernel_source = """
    __kernel void julia(const int width, const int height, const int max_iter,
        const float x_min, const float x_max, const float y_min, const float y_max,
        const float cre, const float cim, __global int* output) {
        int gx = get_global_id(0);
        int gy = get_global_id(1);
        if (gx >= width || gy >= height) return;
        float x0 = x_min + (x_max - x_min) * ((float)gx / (float)(width - 1));
        float y0 = y_min + (y_max - y_min) * ((float)gy / (float)(height - 1));
        float x = x0, y = y0;
        int iter = 0;
        while (x * x + y * y <= 4.0f && iter < max_iter) {
            float xt = x * x - y * y + cre;
            y = 2.0f * x * y + cim;
            x = xt;
            iter++;
        }
        output[gy * width + gx] = iter;
    }
    """
    program = cl.Program(ctx, kernel_source).build()
    result = np.empty(width * height, dtype=np.int32)
    buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, result.nbytes)
    program.julia(
        queue, (width, height), None,
        np.int32(width), np.int32(height), np.int32(max_iter),
        np.float32(x_min), np.float32(x_max), np.float32(y_min), np.float32(y_max),
        np.float32(cre), np.float32(cim), buf,
    )
    cl.enqueue_copy(queue, result, buf)
    queue.finish()
    return result.reshape((height, width))


def generate_burning_ship_opencl(
    width: int = 800,
    height: int = 600,
    max_iter: int = 100,
    x_center: float = -0.5,
    y_center: float = -0.5,
    zoom: float = 1.0,
) -> np.ndarray:
    """Burning Ship su GPU."""
    if not _HAS_OPENCL:
        raise RuntimeError("PyOpenCL non disponibile.")
    aspect_ratio = width / height
    scale = 1.5 / zoom
    x_min = x_center - scale * aspect_ratio
    x_max = x_center + scale * aspect_ratio
    y_min = y_center - scale
    y_max = y_center + scale
    platforms = cl.get_platforms()
    devices = []
    for p in platforms:
        try:
            devices.extend(p.get_devices(device_type=cl.device_type.GPU))
        except Exception:
            continue
    if not devices:
        for p in platforms:
            try:
                devices.extend(p.get_devices())
            except Exception:
                continue
    if not devices:
        raise RuntimeError("Nessun dispositivo OpenCL.")
    ctx = cl.Context(devices=[devices[0]])
    queue = cl.CommandQueue(ctx)
    kernel = """
    __kernel void burning_ship(const int width, const int height, const int max_iter,
        const float x_min, const float x_max, const float y_min, const float y_max,
        __global int* output) {
        int gx = get_global_id(0), gy = get_global_id(1);
        if (gx >= width || gy >= height) return;
        float x0 = x_min + (x_max - x_min) * ((float)gx / (float)(width - 1));
        float y0 = y_min + (y_max - y_min) * ((float)gy / (float)(height - 1));
        float x = 0, y = 0;
        int iter = 0;
        while (x * x + y * y <= 4.0f && iter < max_iter) {
            float ax = fabs(x), ay = fabs(y);
            float xt = ax * ax - ay * ay + x0;
            y = 2.0f * ax * ay + y0;
            x = xt;
            iter++;
        }
        output[gy * width + gx] = iter;
    }
    """
    program = cl.Program(ctx, kernel).build()
    result = np.empty(width * height, dtype=np.int32)
    buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, result.nbytes)
    program.burning_ship(queue, (width, height), None,
        np.int32(width), np.int32(height), np.int32(max_iter),
        np.float32(x_min), np.float32(x_max), np.float32(y_min), np.float32(y_max), buf)
    cl.enqueue_copy(queue, result, buf)
    queue.finish()
    return result.reshape((height, width))


def generate_tricorn_opencl(
    width: int = 800,
    height: int = 600,
    max_iter: int = 100,
    x_center: float = -0.5,
    y_center: float = 0.0,
    zoom: float = 1.0,
) -> np.ndarray:
    """Tricorn su GPU: z = conj(z)^2 + c."""
    if not _HAS_OPENCL:
        raise RuntimeError("PyOpenCL non disponibile.")
    aspect_ratio = width / height
    scale = 1.5 / zoom
    x_min = x_center - scale * aspect_ratio
    x_max = x_center + scale * aspect_ratio
    y_min = y_center - scale
    y_max = y_center + scale
    platforms = cl.get_platforms()
    devices = []
    for p in platforms:
        try:
            devices.extend(p.get_devices(device_type=cl.device_type.GPU))
        except Exception:
            continue
    if not devices:
        for p in platforms:
            try:
                devices.extend(p.get_devices())
            except Exception:
                continue
    if not devices:
        raise RuntimeError("Nessun dispositivo OpenCL.")
    ctx = cl.Context(devices=[devices[0]])
    queue = cl.CommandQueue(ctx)
    kernel = """
    __kernel void tricorn(const int width, const int height, const int max_iter,
        const float x_min, const float x_max, const float y_min, const float y_max,
        __global int* output) {
        int gx = get_global_id(0), gy = get_global_id(1);
        if (gx >= width || gy >= height) return;
        float x0 = x_min + (x_max - x_min) * ((float)gx / (float)(width - 1));
        float y0 = y_min + (y_max - y_min) * ((float)gy / (float)(height - 1));
        float x = 0, y = 0;
        int iter = 0;
        while (x * x + y * y <= 4.0f && iter < max_iter) {
            float xt = x * x - y * y + x0;
            y = -2.0f * x * y + y0;
            x = xt;
            iter++;
        }
        output[gy * width + gx] = iter;
    }
    """
    program = cl.Program(ctx, kernel).build()
    result = np.empty(width * height, dtype=np.int32)
    buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, result.nbytes)
    program.tricorn(queue, (width, height), None,
        np.int32(width), np.int32(height), np.int32(max_iter),
        np.float32(x_min), np.float32(x_max), np.float32(y_min), np.float32(y_max), buf)
    cl.enqueue_copy(queue, result, buf)
    queue.finish()
    return result.reshape((height, width))


def generate_julia(
    width: int = 800,
    height: int = 600,
    max_iter: int = 100,
    x_center: float = 0.0,
    y_center: float = 0.0,
    zoom: float = 1.0,
    c: complex = complex(-0.8, 0.156),
) -> np.ndarray:
    """Calcola un insieme di Julia su CPU con NumPy."""
    x_vals, y_vals = _make_grid(width, height, x_center, y_center, zoom)
    z_vals = x_vals + 1j * y_vals

    iters = np.zeros(z_vals.shape, dtype=int)
    mask = np.ones(z_vals.shape, dtype=bool)

    for i in range(max_iter):
        z_vals[mask] = z_vals[mask] * z_vals[mask] + c
        escaped = np.abs(z_vals) > 2.0
        newly_escaped = escaped & mask
        iters[newly_escaped] = i
        mask &= ~escaped
        if not mask.any():
            break

    iters[mask] = max_iter
    return iters


def generate_burning_ship_cpu(
    width: int = 800,
    height: int = 600,
    max_iter: int = 100,
    x_center: float = -0.5,
    y_center: float = -0.5,
    zoom: float = 1.0,
) -> np.ndarray:
    """Burning Ship: z = (|Re(z)| + i|Im(z)|)^2 + c."""
    x_vals, y_vals = _make_grid(width, height, x_center, y_center, zoom)
    c_vals = x_vals + 1j * y_vals
    z_vals = np.zeros_like(c_vals, dtype=complex)
    iters = np.zeros(c_vals.shape, dtype=int)
    mask = np.ones(c_vals.shape, dtype=bool)
    for i in range(max_iter):
        z_vals[mask] = (np.abs(z_vals[mask].real) + 1j * np.abs(z_vals[mask].imag)) ** 2 + c_vals[mask]
        escaped = np.abs(z_vals) > 2.0
        newly_escaped = escaped & mask
        iters[newly_escaped] = i
        mask &= ~escaped
        if not mask.any():
            break
    iters[mask] = max_iter
    return iters


def generate_tricorn_cpu(
    width: int = 800,
    height: int = 600,
    max_iter: int = 100,
    x_center: float = -0.5,
    y_center: float = 0.0,
    zoom: float = 1.0,
) -> np.ndarray:
    """Tricorn: z = conj(z)^2 + c."""
    x_vals, y_vals = _make_grid(width, height, x_center, y_center, zoom)
    c_vals = x_vals + 1j * y_vals
    z_vals = np.zeros_like(c_vals, dtype=complex)
    iters = np.zeros(c_vals.shape, dtype=int)
    mask = np.ones(c_vals.shape, dtype=bool)
    for i in range(max_iter):
        z_vals[mask] = np.conj(z_vals[mask]) ** 2 + c_vals[mask]
        escaped = np.abs(z_vals) > 2.0
        newly_escaped = escaped & mask
        iters[newly_escaped] = i
        mask &= ~escaped
        if not mask.any():
            break
    iters[mask] = max_iter
    return iters


def generate_newton_cpu(
    width: int = 800,
    height: int = 600,
    max_iter: int = 50,
    x_center: float = 0.0,
    y_center: float = 0.0,
    zoom: float = 1.0,
) -> np.ndarray:
    """Newton per z^3-1: z = z - (z^3-1)/(3z^2). Colore = iterazioni per convergere."""
    x_vals, y_vals = _make_grid(width, height, x_center, y_center, zoom)
    z_vals = x_vals + 1j * y_vals
    iters = np.zeros(z_vals.shape, dtype=int)
    for i in range(max_iter):
        f = z_vals**3 - 1
        fp = 3 * z_vals**2
        z_new = z_vals - f / (fp + 1e-10)
        converged = np.abs(z_new - z_vals) < 1e-6
        iters[~converged] = i
        z_vals = z_new
        if converged.all():
            break
    return iters


# Colormap disponibili (Matplotlib)
_COLORMAPS = [
    "inferno",
    "viridis",
    "plasma",
    "magma",
    "turbo",
    "hot",
    "cool",
    "winter",
    "spring",
    "summer",
    "autumn",
    "twilight",
    "Spectral",
    "RdYlBu",
]


class FractalApp:
    """Applicazione interattiva per esplorare frattali Mandelbrot/Julia."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.width = args.width
        self.height = args.height
        self.max_iter = args.max_iter
        self.backend = args.backend
        self.fractal_type = args.type
        self.x_center = args.x_center
        self.y_center = args.y_center
        self.zoom = args.zoom
        self.julia_c = complex(args.julia_cre, args.julia_cim)
        try:
            self.cmap_index = _COLORMAPS.index(args.cmap)
        except ValueError:
            self.cmap_index = 0
        self.smooth_coloring = getattr(args, "smooth", False)
        self.zoom_history: list[tuple[float, float, float]] = []
        self.fullscreen = getattr(args, "fullscreen", False)

        self.fig = plt.figure(figsize=(11, 6))

        self.fig = plt.figure(figsize=(11, 6))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[3, 1])
        self.ax = self.fig.add_subplot(gs[0])
        self.ax_legend = self.fig.add_subplot(gs[1])
        self.image = None
        self.colorbar = None

        self.fig = plt.figure(figsize=(11, 6))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[3, 1])
        self.ax = self.fig.add_subplot(gs[0])
        self.ax_legend = self.fig.add_subplot(gs[1])

        self._draw_initial()
        self._connect_events()
        if self.fullscreen:
            self.fig.canvas.manager.window.state("zoomed")

    def _next_save_number(self) -> int:
        """Restituisce il prossimo numero per il salvataggio automatico."""
        pattern = os.path.join(os.getcwd(), "fractal_*.png")
        existing = glob.glob(pattern)
        numbers = []
        for p in existing:
            m = re.search(r"fractal_(\d+)\.png$", os.path.basename(p))
            if m:
                numbers.append(int(m.group(1)))
        return max(numbers, default=0) + 1

    def _save_current(self, path: str | None = None) -> None:
        """Salva l'immagine corrente del frattale su disco (solo area frattale, senza legenda)."""
        if path is None:
            num = self._next_save_number()
            filename = f"fractal_{num:03d}.png"
        else:
            filename = path

        self.fig.canvas.draw()
        extent = self.ax.get_window_extent().transformed(
            self.fig.dpi_scale_trans.inverted()
        )
        self.fig.savefig(filename, bbox_inches=extent, pad_inches=0)
        print(f"Immagine salvata in: {filename}")

    def _save_current_highres(self, path: str | None = None) -> None:
        """Salva in alta risoluzione (4K)."""
        if path is None:
            num = self._next_save_number()
            filename = f"fractal_{num:03d}_4k.png"
        else:
            filename = path
        w, h = 3840, 2160
        common = dict(
            width=w, height=h, max_iter=self.max_iter,
            x_center=self.x_center, y_center=self.y_center, zoom=self.zoom,
        )
        def _gen():
            c = dict(**common)
            if self.fractal_type == "mandelbrot":
                return generate_mandelbrot_opencl(**c) if self.backend == "opencl" else generate_mandelbrot_cpu(**c)
            if self.fractal_type == "julia":
                return generate_julia_opencl(**c, c=self.julia_c) if self.backend == "opencl" else generate_julia(**c, c=self.julia_c)
            if self.fractal_type == "burning_ship":
                return generate_burning_ship_opencl(**c) if self.backend == "opencl" else generate_burning_ship_cpu(**c)
            if self.fractal_type == "tricorn":
                return generate_tricorn_opencl(**c) if self.backend == "opencl" else generate_tricorn_cpu(**c)
            return generate_newton_cpu(**c)
        data = _gen()
        fig_hires = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        ax_hires = fig_hires.add_subplot(111)
        ax_hires.imshow(data, cmap=self._get_cmap(), origin="lower")
        ax_hires.axis("off")
        fig_hires.savefig(filename, bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close(fig_hires)
        print(f"Immagine 4K salvata in: {filename}")

    def _animate_zoom(self) -> None:
        """Genera animazione zoom e salva come GIF."""
        try:
            import imageio
        except ImportError:
            print("Installa imageio: pip install imageio")
            return
        num = self._next_save_number()
        filename = f"fractal_anim_{num:03d}.gif"
        frames = 30
        z_start, z_end = self.zoom, self.zoom * 10
        common = dict(
            width=640, height=480, max_iter=self.max_iter,
            x_center=self.x_center, y_center=self.y_center,
        )
        def gen():
            for i in range(frames):
                t = i / (frames - 1)
                zoom = z_start * (z_end / z_start) ** t
                if self.fractal_type == "mandelbrot":
                    d = generate_mandelbrot_opencl(**common, zoom=zoom) if self.backend == "opencl" else generate_mandelbrot_cpu(**common, zoom=zoom)
                elif self.fractal_type == "julia":
                    d = generate_julia_opencl(**common, zoom=zoom, c=self.julia_c) if self.backend == "opencl" else generate_julia(**common, zoom=zoom, c=self.julia_c)
                elif self.fractal_type == "burning_ship":
                    d = generate_burning_ship_opencl(**common, zoom=zoom) if self.backend == "opencl" else generate_burning_ship_cpu(**common, zoom=zoom)
                elif self.fractal_type == "tricorn":
                    d = generate_tricorn_opencl(**common, zoom=zoom) if self.backend == "opencl" else generate_tricorn_cpu(**common, zoom=zoom)
                else:
                    d = generate_newton_cpu(**common, zoom=zoom)
                cmap = plt.get_cmap(self._get_cmap())
                norm = plt.Normalize(vmin=0, vmax=d.max())
                img = (cmap(norm(d))[:, :, :3] * 255).astype(np.uint8)
                yield img
        print("Generazione animazione...")
        imageio.mimsave(filename, list(gen()), duration=0.1, loop=0)
        print(f"Animazione salvata in: {filename}")

    def _compute(self) -> tuple[np.ndarray, str]:
        common = dict(
            width=self.width,
            height=self.height,
            max_iter=self.max_iter,
            x_center=self.x_center,
            y_center=self.y_center,
            zoom=self.zoom,
        )
        if self.fractal_type == "mandelbrot":
            if self.backend == "opencl":
                data = generate_mandelbrot_opencl(**common)
            else:
                data = generate_mandelbrot_cpu(**common)
            title = f"Mandelbrot [{self.backend}] (z={self.zoom:.2f})"
        elif self.fractal_type == "julia":
            if self.backend == "opencl":
                try:
                    data = generate_julia_opencl(**common, c=self.julia_c)
                except Exception:
                    data = generate_julia(**common, c=self.julia_c)
            else:
                data = generate_julia(**common, c=self.julia_c)
            title = f"Julia c={self.julia_c.real:.3f}+{self.julia_c.imag:.3f}i [{self.backend}]"
        elif self.fractal_type == "burning_ship":
            if self.backend == "opencl":
                data = generate_burning_ship_opencl(**common)
            else:
                data = generate_burning_ship_cpu(**common)
            title = f"Burning Ship [{self.backend}] (z={self.zoom:.2f})"
        elif self.fractal_type == "tricorn":
            if self.backend == "opencl":
                data = generate_tricorn_opencl(**common)
            else:
                data = generate_tricorn_cpu(**common)
            title = f"Tricorn [{self.backend}] (z={self.zoom:.2f})"
        elif self.fractal_type == "newton":
            data = generate_newton_cpu(**common)
            title = f"Newton z³-1 (z={self.zoom:.2f})"
        else:
            data = generate_mandelbrot_cpu(**common)
            title = f"Mandelbrot (z={self.zoom:.2f})"
        return data, title

    def _get_cmap(self) -> str:
        """Restituisce la colormap corrente."""
        return _COLORMAPS[self.cmap_index % len(_COLORMAPS)]

    def _draw_initial(self) -> None:
        data, title = self._compute()
        self.image = self.ax.imshow(data, cmap=self._get_cmap(), origin="lower")
        self.colorbar = self.fig.colorbar(self.image, ax=self.ax, label="Iterazioni")
        self.ax.set_title(title)

        # Legenda comandi nel pannello laterale
        self.ax_legend.axis("off")
        legend_text = (
            "TIPO FRATTALE\n"
            "  m = Mandelbrot  j = Julia\n"
            "  b = Burning Ship  t = Tricorn\n"
            "  n = Newton\n\n"
            "VISTA\n"
            "  + = zoom avanti\n"
            "  - = zoom indietro\n"
            "  frecce = sposta\n"
            "  click = centra qui\n"
            "  r = reset vista\n\n"
            "PARAMETRI\n"
            "  [ = meno iterazioni\n"
            "  ] = più iterazioni\n"
            "  c = cambia colori\n\n"
            "SALVA\n"
            "  s = salva   S = 4K   a = anim\n\n"
            "SOLO JULIA\n"
            "  1-4 = Re/Im(c)\n"
            "  5-9 = preset (dendrite, spirale...)\n\n"
            "  u = indietro zoom\n"
            "  x = copia parametri"
        )
        self.ax_legend.text(
            0.05, 0.5, legend_text,
            transform=self.ax_legend.transAxes,
            fontsize=8,
            verticalalignment="center",
            family="monospace",
        )

        self.fig.tight_layout()

        self._print_help()

    def _draw_3d(self) -> None:
        """Disegna il frattale in 3D come superficie."""
        data, title = self._compute()
        
        # Downsampling for 3D performance (use fewer points)
        step = max(1, min(self.width, self.height) // 200)
        x_data = np.arange(0, self.width, step)
        y_data = np.arange(0, self.height, step)
        X, Y = np.meshgrid(x_data, y_data)
        Z = data[::step, ::step]
        
        # Normalizza Z per una migliore visualizzazione
        Z_norm = np.log1p(Z)  # log(1+Z) per migliorare la visualizzazione
        
        cmap = plt.get_cmap(self._get_cmap())
        norm = plt.Normalize(vmin=Z_norm.min(), vmax=Z_norm.max())
        colors = cmap(norm(Z_norm))
        
        self.surf = self.ax.plot_surface(
            X, Y, Z_norm,
            facecolors=colors,
            linewidth=0,
            antialiased=True,
            shade=True
        )
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Iterazioni (log)')
        self.ax.set_title(f'{title} - Visualizzazione 3D')
        
        # Legenda 3D
        legend_text = (
            "VISUALIZZAZIONE 3D\n"
            "  Il frattale è rappresentato\n"
            "  come superficie 3D dove\n"
            "  l'altezza = iterazioni\n\n"
            "CONTROLLI MOUSE\n"
            "  click + drag = ruota\n"
            "  scroll = zoom\n"
            "  right click = pan\n\n"
            "TASTIERA\n"
            "  m = Mandelbrot\n"
            "  j = Julia\n"
            "  b = Burning Ship\n"
            "  t = Tricorn\n"
            "  n = Newton\n"
            "  + - = zoom\n"
            "  c = cambia colori\n"
            "  r = reset vista\n"
            "  v =切换 2D/3D"
        )
        self.ax.text2D(0.02, 0.98, legend_text, transform=self.ax.transAxes,
                       fontsize=8, verticalalignment='top', family='monospace')
        
        self._print_help_3d()

    def _print_help_3d(self) -> None:
        """Stampa i comandi 3D nel terminale."""
        help_text = """
╔══════════════════════════════════════════════════════════════╗
║  CONTROLLI 3D                                                 ║
╠══════════════════════════════════════════════════════════════╣
║  Mouse: click+drag=ruota, scroll=zoom, right=pan             ║
║  m=Mandelbrot j=Julia b=BurningShip t=Tricorn n=Newton        ║
║  + - zoom   c colori   r reset   v = passa a 2D               ║
╚══════════════════════════════════════════════════════════════╝
"""
        print(help_text)

    def _connect_events_3d(self) -> None:
        """Connette gli eventi per la modalità 3D."""
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_3d)

    def _print_help(self) -> None:
        """Stampa i comandi disponibili nel terminale."""
        help_text = """
╔══════════════════════════════════════════════════════════════╗
║  CONTROLLI FRATTALE                                          ║
╠══════════════════════════════════════════════════════════════╣
║  m=Mandelbrot j=Julia b=BurningShip t=Tricorn n=Newton        ║
║  + - zoom   frecce sposta   click centra   r reset             ║
║  [ ] iter   c colori   s salva   S 4K   a anim   u indietro   ║
║  x copia params   Julia: 1-4 Re/Im   5-9 preset                ║
╚══════════════════════════════════════════════════════════════╝
"""
        print(help_text)

    def _connect_events(self) -> None:
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _update_view(self) -> None:
        data, title = self._compute()
        self.image.set_data(data)
        self.image.set_cmap(self._get_cmap())
        self.ax.set_title(title)
        self.fig.canvas.draw_idle()

    def _update_view_3d(self) -> None:
        """Aggiorna la visualizzazione 3D."""
        data, title = self._compute()
        
        # Downsampling for 3D performance
        step = max(1, min(self.width, self.height) // 200)
        x_data = np.arange(0, self.width, step)
        y_data = np.arange(0, self.height, step)
        X, Y = np.meshgrid(x_data, y_data)
        Z = data[::step, ::step]
        Z_norm = np.log1p(Z)
        
        cmap = plt.get_cmap(self._get_cmap())
        norm = plt.Normalize(vmin=Z_norm.min(), vmax=Z_norm.max())
        colors = cmap(norm(Z_norm))
        
        # Rimuovi la superficie precedente senza usare clear()
        if hasattr(self, 'surf') and self.surf is not None:
            self.surf.remove()
        
        self.surf = self.ax.plot_surface(
            X, Y, Z_norm,
            facecolors=colors,
            linewidth=0,
            antialiased=True,
            shade=True
        )
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Iterazioni (log)')
        self.ax.set_title(f'{title} - Visualizzazione 3D')
        
        legend_text = (
            "VISUALIZZAZIONE 3D\n"
            "  Il frattale è rappresentato\n"
            "  come superficie 3D dove\n"
            "  l'altezza = iterazioni\n\n"
            "CONTROLLI MOUSE\n"
            "  click + drag = ruota\n"
            "  scroll = zoom\n"
            "  right click = pan\n\n"
            "TASTIERA\n"
            "  m = Mandelbrot\n"
            "  j = Julia\n"
            "  b = Burning Ship\n"
            "  t = Tricorn\n"
            "  n = Newton\n"
            "  + - = zoom\n"
            "  c = cambia colori\n"
            "  r = reset vista\n"
            "  v = passa a 2D"
        )
        self.ax.text2D(0.02, 0.98, legend_text, transform=self.ax.transAxes,
                       fontsize=8, verticalalignment='top', family='monospace')
        
        self.fig.canvas.draw_idle()

    def _on_scroll(self, event) -> None:
        if event.inaxes != self.ax:
            return
        self._push_zoom()
        factor = 0.8 if event.button == "up" else 1.25
        self.zoom *= 1 / factor
        self._update_view()

    def _on_click(self, event) -> None:
        if event.inaxes != self.ax or event.button != 1:
            return
        self._push_zoom()

        # Mappa la coordinata di click alle coordinate del piano complesso approssimando
        # all'intervallo standard usato da _make_grid.
        base_scale = 1.5 / self.zoom
        aspect_ratio = self.width / self.height
        x_min = self.x_center - base_scale * aspect_ratio
        x_max = self.x_center + base_scale * aspect_ratio
        y_min = self.y_center - base_scale
        y_max = self.y_center + base_scale

        if event.xdata is not None and event.ydata is not None:
            # event.xdata/ydata sono in coordinate di immagine (0..width, 0..height)
            self.x_center = x_min + (x_max - x_min) * (event.xdata / self.width)
            self.y_center = y_min + (y_max - y_min) * (event.ydata / self.height)
            self._update_view()

    def _update_colormap_only(self) -> None:
        """Aggiorna solo la colormap senza ricalcolare il frattale."""
        self.image.set_cmap(self._get_cmap())
        self.fig.canvas.draw_idle()

    def _push_zoom(self) -> None:
        """Salva lo stato corrente per 'indietro'."""
        self.zoom_history.append((self.x_center, self.y_center, self.zoom))

    def _pop_zoom(self) -> bool:
        """Ripristina lo stato precedente. Ritorna True se fatto."""
        if not self.zoom_history:
            return False
        self.x_center, self.y_center, self.zoom = self.zoom_history.pop()
        return True

    def _copy_params(self) -> None:
        """Copia i parametri negli appunti o stampa."""
        s = (
            f"--type {self.fractal_type} --x-center {self.x_center:.6f} "
            f"--y-center {self.y_center:.6f} --zoom {self.zoom:.6f} "
            f"--max-iter {self.max_iter}"
        )
        if self.fractal_type == "julia":
            s += f" --julia-cre {self.julia_c.real:.6f} --julia-cim {self.julia_c.imag:.6f}"
        try:
            if sys.platform == "win32":
                subprocess.run("clip", input=s.encode(), shell=True, check=True)
                print("Parametri copiati negli appunti.")
            else:
                subprocess.run(["xclip", "-selection", "clipboard"], input=s.encode(), check=True)
                print("Parametri copiati negli appunti.")
        except Exception:
            print("Parametri (copia manuale):\n", s)

    def _on_key(self, event) -> None:
        if event.key == "m":
            self.fractal_type = "mandelbrot"
        elif event.key == "j":
            self.fractal_type = "julia"
        elif event.key == "b":
            self.fractal_type = "burning_ship"
        elif event.key == "t":
            self.fractal_type = "tricorn"
        elif event.key == "n":
            self.fractal_type = "newton"
        elif event.key == "s":
            self._save_current()
            return
        elif event.key == "S":
            self._save_current_highres()
            return
        elif event.key == "a":
            self._animate_zoom()
            return
        elif event.key == "x":
            self._copy_params()
            return
        elif event.key == "u":
            if self._pop_zoom():
                self._update_view()
            else:
                print("Nessuna cronologia zoom.")
            return
        elif event.key == "c":
            self.cmap_index += 1
            self._update_colormap_only()
            print(f"Colormap: {self._get_cmap()}")
            return
        elif event.key in {"+", "add"}:
            self._push_zoom()
            self.zoom *= 1.25
        elif event.key in {"-", "subtract"}:
            self._push_zoom()
            self.zoom /= 1.25
        elif event.key == "[":
            self.max_iter = max(10, self.max_iter - 50)
            print(f"max_iter = {self.max_iter}")
        elif event.key == "]":
            self.max_iter = min(5000, self.max_iter + 50)
            print(f"max_iter = {self.max_iter}")
        elif event.key == "r":
            defaults = {
                "mandelbrot": (-0.5, 0.0),
                "julia": (0.0, 0.0),
                "burning_ship": (-0.5, -0.5),
                "tricorn": (-0.5, 0.0),
                "newton": (0.0, 0.0),
            }
            self.x_center, self.y_center = defaults.get(self.fractal_type, (-0.5, 0.0))
            self.zoom = 1.0
        elif event.key == "up":
            self._push_zoom()
            self.y_center += 0.2 / self.zoom
        elif event.key == "down":
            self._push_zoom()
            self.y_center -= 0.2 / self.zoom
        elif event.key == "left":
            self._push_zoom()
            self.x_center -= 0.2 / self.zoom
        elif event.key == "right":
            self._push_zoom()
            self.x_center += 0.2 / self.zoom
        elif event.key == "v":
            print("Per passare a 3D, esci e riavvia con: python fractal_viewer.py --mode 3d")
            return
        elif self.fractal_type == "julia":
            delta = 0.05
            if event.key in ("1", "numpad1"):
                self.julia_c = complex(self.julia_c.real - delta, self.julia_c.imag)
                print(f"Julia c = {self.julia_c.real:.3f}+{self.julia_c.imag:.3f}i")
            elif event.key in ("2", "numpad2"):
                self.julia_c = complex(self.julia_c.real + delta, self.julia_c.imag)
                print(f"Julia c = {self.julia_c.real:.3f}+{self.julia_c.imag:.3f}i")
            elif event.key in ("3", "numpad3"):
                self.julia_c = complex(self.julia_c.real, self.julia_c.imag - delta)
                print(f"Julia c = {self.julia_c.real:.3f}+{self.julia_c.imag:.3f}i")
            elif event.key in ("4", "numpad4"):
                self.julia_c = complex(self.julia_c.real, self.julia_c.imag + delta)
                print(f"Julia c = {self.julia_c.real:.3f}+{self.julia_c.imag:.3f}i")
            elif event.key in ("5", "6", "7", "8", "9"):
                presets = list(JULIA_PRESETS.values())
                idx = int(event.key) - 5
                if idx < len(presets):
                    re_c, im_c = presets[idx]
                    self.julia_c = complex(re_c, im_c)
                    name = list(JULIA_PRESETS.keys())[idx]
                    print(f"Julia preset: {name} c={self.julia_c.real:.3f}+{self.julia_c.imag:.3f}i")
            else:
                return
        else:
            return

        self._update_view()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualizza diversi tipi di frattali.")
    parser.add_argument(
        "--type",
        choices=["mandelbrot", "julia", "burning_ship", "tricorn", "newton"],
        default="mandelbrot",
        help="Tipo di frattale da generare.",
    )
    parser.add_argument("--width", type=int, default=800, help="Larghezza immagine in pixel.")
    parser.add_argument("--height", type=int, default=600, help="Altezza immagine in pixel.")
    parser.add_argument("--max-iter", type=int, default=200, help="Numero massimo di iterazioni.")
    parser.add_argument("--x-center", type=float, default=-0.5, help="Centro orizzontale.")
    parser.add_argument("--y-center", type=float, default=0.0, help="Centro verticale.")
    parser.add_argument("--zoom", type=float, default=1.0, help="Fattore di zoom (1 = standard).")
    parser.add_argument(
        "--backend",
        choices=["cpu", "opencl"],
        default="cpu",
        help=(
            "Backend di calcolo: 'cpu' usa NumPy, 'opencl' usa la GPU "
            "tramite OpenCL (se disponibile)."
        ),
    )

    parser.add_argument(
        "--julia-cre",
        type=float,
        default=-0.8,
        help="Parte reale del parametro c per Julia.",
    )
    parser.add_argument(
        "--julia-cim",
        type=float,
        default=0.156,
        help="Parte immaginaria del parametro c per Julia.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="inferno",
        choices=_COLORMAPS,
        help="Colormap iniziale per la visualizzazione.",
    )
    parser.add_argument("--fullscreen", action="store_true", help="Avvia in fullscreen.")
    parser.add_argument("--smooth", action="store_true", help="Smooth coloring (sperimentale).")

    args = parser.parse_args()

    app = FractalApp(args)
    plt.show()


if __name__ == "__main__":
    main()

