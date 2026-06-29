import contextlib
import os
import sys


_TENSORRT_DLL_DIRECTORY_HANDLES = []


def prepare_tensorrt_import():
    if not sys.platform.startswith("win"):
        return

    import sysconfig

    libs = os.path.join(sysconfig.get_paths().get("purelib", ""), "tensorrt_libs")
    if os.path.isdir(libs):
        os.environ["PATH"] = libs + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                handle = os.add_dll_directory(libs)
                _TENSORRT_DLL_DIRECTORY_HANDLES.append(handle)
            except OSError:
                pass

    try:
        import tensorrt  # noqa: F401
    except ImportError:
        try:
            import tensorrt_bindings as tensorrt
        except ImportError:
            return
        sys.modules.setdefault("tensorrt", tensorrt)

    tensorrt_dir = os.path.dirname(sys.modules["tensorrt"].__file__)
    candidate_dirs = [
        tensorrt_dir,
        os.path.join(tensorrt_dir, "libs"),
        os.path.join(os.path.dirname(tensorrt_dir), "tensorrt_libs"),
    ]
    for directory in candidate_dirs:
        if os.path.isdir(directory):
            os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    handle = os.add_dll_directory(directory)
                    _TENSORRT_DLL_DIRECTORY_HANDLES.append(handle)
                except OSError:
                    pass


@contextlib.contextmanager
def quiet_terminal_output():
    try:
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                yield
        return

    sys.stdout.flush()
    sys.stderr.flush()
    with open(os.devnull, "w") as devnull:
        saved_stdout_fd = os.dup(stdout_fd)
        saved_stderr_fd = os.dup(stderr_fd)
        try:
            os.dup2(devnull.fileno(), stdout_fd)
            os.dup2(devnull.fileno(), stderr_fd)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout_fd, stdout_fd)
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)


def select_torch_device(torch):
    """
    Pick the best available torch device.

    Order:
      1) CUDA (NVIDIA)
      2) DirectML (Windows, AMD/Intel/NVIDIA via torch-directml) if installed
      3) CPU

    Returns: (device_obj, label, hint)
    """
    hint = None

    try:
        if torch.cuda.is_available():
            return torch.device("cuda"), "CUDA", None
    except Exception:
        pass

    if sys.platform.startswith("win"):
        try:
            import torch_directml  # type: ignore

            return torch_directml.device(), "DirectML", None
        except Exception:
            hint = (
                "Kein CUDA/DirectML-Backend verfuegbar, falle auf CPU zurueck. "
                "NVIDIA: installiere ein CUDA-faehiges PyTorch (siehe https://pytorch.org). "
                "AMD/Intel (Windows): 'pip install torch-directml' probieren."
            )

    return torch.device("cpu"), "CPU", hint
