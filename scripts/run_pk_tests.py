#!/usr/bin/env python
"""Run the P(k) test suite and print a summary.

Usage:
    python scripts/run_pk_tests.py [--fast] [--full] [--all]

Modes:
    --fast  Run only fast-mode tests (default, ~2-5 min on GPU)
    --full  Run only full-mode tests (slow, ~15-30 min on GPU)
    --all   Run both fast and full modes

Runs pytest on the P(k)-related test files and prints a compact summary.
Exit code is 0 if all tests pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


TEST_FILES = [
    "tests/test_pk_accuracy.py",
    "tests/test_pk_gradients.py",
    "tests/test_perturbations.py",
]


def main():
    parser = argparse.ArgumentParser(description="Run P(k) test suite on GPU")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true", default=True,
                      help="Run fast-mode tests only (default)")
    mode.add_argument("--full", action="store_true",
                      help="Run full-mode tests only")
    mode.add_argument("--all", action="store_true",
                      help="Run both fast and full modes")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose pytest output")
    args = parser.parse_args()

    # Verify GPU is available
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform == "gpu"]
        print(f"JAX devices: {devices}")
        if gpu_devices:
            print(f"GPU available: {gpu_devices[0]}")
        else:
            print("WARNING: No GPU detected, tests will run on CPU (slow)")
    except ImportError:
        print("ERROR: JAX not importable", file=sys.stderr)
        sys.exit(1)

    # Build pytest command
    pytest_args = ["python", "-m", "pytest"]
    pytest_args.extend(TEST_FILES)
    pytest_args.extend(["-x", "--tb=short"])

    if args.verbose:
        pytest_args.append("-v")
    else:
        pytest_args.append("-q")

    if args.all:
        # Run without --fast flag (pytest runs both fast and full)
        pass
    elif args.full:
        # Run without --fast: full mode only, skip fast-only tests
        pass
    else:
        # Default: fast mode
        pytest_args.append("--fast")

    print(f"\n{'='*60}")
    print(f"P(k) test suite")
    print(f"Command: {' '.join(pytest_args)}")
    print(f"{'='*60}\n")

    t0 = time.time()
    result = subprocess.run(pytest_args)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"Completed in {elapsed:.1f}s")
    print(f"Exit code: {result.returncode}")
    print(f"{'='*60}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
