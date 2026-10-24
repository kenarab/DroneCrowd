"""Diagnóstico sin descargas; prueba CUDA opcional de cuatro valores."""
import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import sys

PACKAGES = ("torch", "torchvision", "numpy", "scipy", "scikit-learn",
            "pillow", "h5py", "opencv-python", "spatial-correlation-sampler")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--gpu-smoke", action="store_true")
    args = parser.parse_args()
    expected = Path("/home/alephbet/home.agents/tmp").resolve()
    for name in ("TMPDIR", "TMP", "TEMP"):
        value = os.environ.get(name)
        if not value or Path(value).resolve() != expected:
            parser.error("Ejecutar mediante ops/python-ml.sh: temporal autorizado obligatorio")
    report = {"python": sys.version.split()[0], "executable": sys.executable,
              "packages": {}, "runtime_checked": False, "gpu_smoke": False,
              "model_inference": False, "training": False}
    for name in PACKAGES:
        try:
            report["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            report["packages"][name] = None
    status = 0
    if args.runtime or args.gpu_smoke:
        try:
            import torch
            report["runtime_checked"] = True
            report["torch_runtime"] = torch.__version__
            report["torch_cuda"] = torch.version.cuda
            report["cuda_available"] = torch.cuda.is_available()
            if args.gpu_smoke:
                if not report["cuda_available"]:
                    raise RuntimeError("CUDA no accesible; comprobar aislamiento antes del driver")
                with torch.no_grad():
                    value = (torch.arange(4, device="cuda") * 2).sum().item()
                torch.cuda.synchronize()
                report["gpu_smoke"] = value == 12
                report["gpu_name"] = torch.cuda.get_device_name(0)
                if value != 12:
                    raise RuntimeError("Resultado sintético inesperado")
        except Exception as error:
            report["runtime_error"] = type(error).__name__ + ": " + str(error)
            status = 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
