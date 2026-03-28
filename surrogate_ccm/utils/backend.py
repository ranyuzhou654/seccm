"""GPU/CPU backend abstraction for array operations."""

import numpy as np


def gpu_available():
    """Check if CuPy and a CUDA GPU are available."""
    try:
        import cupy
        cupy.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def get_array_module(use_gpu=True):
    """Return cupy if GPU is requested and available, else numpy.

    Parameters
    ----------
    use_gpu : bool
        Whether to attempt GPU acceleration.

    Returns
    -------
    xp : module
        Either ``cupy`` or ``numpy``.
    """
    if use_gpu and gpu_available():
        import cupy
        return cupy
    return np


def to_device(arr, xp):
    """Transfer a numpy array to the target backend.

    Parameters
    ----------
    arr : ndarray
        Input array (numpy).
    xp : module
        Target array module (numpy or cupy).

    Returns
    -------
    out : ndarray
        Array on the target device.
    """
    if xp is np:
        return np.asarray(arr)
    return xp.asarray(arr)


def to_numpy(arr):
    """Ensure array is a numpy array (transfer from GPU if needed).

    Parameters
    ----------
    arr : ndarray
        Input array (numpy or cupy).

    Returns
    -------
    out : numpy.ndarray
    """
    if isinstance(arr, np.ndarray):
        return arr
    # CuPy array
    return arr.get()
