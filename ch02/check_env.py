"""2부 실습 — 개발환경 확인

Python / PyTorch / CUDA / MuJoCo / OpenCV 가 제대로 설치되었는지 한 번에 확인합니다.

실행:
    python ch02/check_env.py
"""

import platform
import sys


def check(name, fn):
    try:
        print(f"{name:<12} {fn()}")
    except Exception as e:  # noqa: BLE001
        print(f"{name:<12} 없음 ({type(e).__name__}: {e})")


def torch_info():
    import torch

    line = f"v{torch.__version__}  CUDA 사용 가능: {torch.cuda.is_available()}"
    if torch.cuda.is_available():
        line += f"  GPU: {torch.cuda.get_device_name(0)}"
        line += f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB"
    return line


if __name__ == "__main__":
    print(f"{'OS':<12} {platform.platform()}")
    print(f"{'Python':<12} {sys.version.split()[0]}")
    check("NumPy", lambda: "v" + __import__("numpy").__version__)
    check("PyTorch", torch_info)
    check("MuJoCo", lambda: "v" + __import__("mujoco").__version__)
    check("OpenCV", lambda: "v" + __import__("cv2").__version__)
    check("imageio", lambda: "v" + __import__("imageio").__version__)
