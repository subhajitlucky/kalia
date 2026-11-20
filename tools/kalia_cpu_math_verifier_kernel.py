"""Kaggle CPU pilot for the math verifier.

The batch TPU queue is server-side and can take hours. This runs the same
JAX/Flax verifier on Kaggle CPU with a smaller synthetic set, so the pipeline
and metrics are validated while the TPU job waits.
"""

import sys
from pathlib import Path

script = next(Path("/kaggle/input").glob("**/kalia_tpu_math_verifier_kernel.py"))
sys.argv = [
    str(script),
    "--allow-cpu",
    "--train-n", "20000",
    "--val-n", "2000",
    "--epochs", "2",
    "--batch-size", "128",
]
code = compile(script.read_text(), str(script), "exec")
exec(code, {"__name__": "__main__"})
