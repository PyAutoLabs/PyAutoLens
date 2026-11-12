(installation-acceleration)=

# GPU and optional packages

## JAX and GPUs

The standard installation includes JAX on platforms supported by the package’s dependency markers. A GPU is not required.

For GPU use, follow the [official JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for your hardware, in the same environment as PyAutoLens. Keep JAX compatible with the requirements of your installed PyAuto packages, then check:

```bash
python -m pip check
python -c "import jax; print(jax.devices())"
```

The device list should include your GPU. If it lists only CPU devices, follow JAX’s platform-specific troubleshooting instructions. Native Windows does not support JAX’s NVIDIA GPU installation; consult its WSL2 guidance instead.

On Intel Macs, the PyAuto package markers omit JAX and use the NumPy path. GPU acceleration is not enabled by the standard macOS install.

## Numba

Numba accelerates selected CPU calculations. Install it if your workflow uses those calculations or an example requests it:

```bash
python -m pip install numba
```

It is optional; it is not a prerequisite for every linear-profile or pixelization fit.

## Interferometer calculations

Some interferometer workflows require `nufftax`, particularly the JAX NUFFT path:

```bash
python -m pip install nufftax
python -m pip check
```

The NumPy NUFFT path uses `finufft` on platforms where it is supplied by the package dependencies. Follow the chosen example’s requirements rather than installing every optional package in advance.
