"""Zero-dependency mutation testing runner for core Parallax modules.

Introduces controlled, deliberate mutations (arithmetic operator flips,
constant modifications, relational alterations) into core modules to verify
that existing unit tests detect regressions and faults.
"""

import os
import subprocess
import sys
from typing import Dict, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MUTATION_SPECS: List[Dict[str, object]] = [
    {
        "file": "parallax/tensor.py",
        "target": "out._backward = _backward",
        "mutant": "pass",
        "test": "tests/test_tensor.py",
        "description": "Disable Tensor backward function binding",
    },
    {
        "file": "parallax/loss.py",
        "target": 'out = Tensor(mean_loss, (logits,), "CrossEntropyLoss")',
        "mutant": 'out = Tensor(mean_loss * 0.0, (logits,), "CrossEntropyLoss")',
        "test": "tests/test_loss.py",
        "description": "Zero out cross entropy loss scalar",
    },
    {
        "file": "parallax/optim.py",
        "target": "p.data -= step_update",
        "mutant": "p.data += step_update",
        "test": "tests/test_train.py",
        "description": "Invert AdamW parameter update direction",
    },
    {
        "file": "parallax/attention.py",
        "target": "scores = np.where(causal_mask[None, None, :, :], -1e9, scores)",
        "mutant": "pass",
        "test": "tests/test_attention.py",
        "description": "Disable causal attention upper-triangular masking",
    },
    {
        "file": "parallax/generator.py",
        "target": "scaled_logits = logits.copy() / temperature",
        "mutant": "scaled_logits = logits.copy() * temperature",
        "test": "tests/test_kto.py",
        "description": "Invert temperature scaling in generator",
    },
]


def run_mutation_test() -> int:
    """Run mutation tests and report kill score."""
    killed = 0
    survived = 0
    total = len(MUTATION_SPECS)

    print(f"Running mutation testing across {total} targets...\n")

    for idx, spec in enumerate(MUTATION_SPECS, 1):
        rel_file = str(spec["file"])
        file_path = os.path.join(REPO_ROOT, rel_file)
        target = str(spec["target"])
        mutant = str(spec["mutant"])
        test_file = str(spec["test"])
        description = str(spec["description"])

        with open(file_path, "r", encoding="utf-8") as f:
            original_content = f.read()

        if target not in original_content:
            print(f"[{idx}/{total}] SKIP: Target not found in {rel_file}")
            continue

        mutated_content = original_content.replace(target, mutant, 1)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(mutated_content)

            cmd = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                os.path.join(REPO_ROOT, test_file),
            ]
            res = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
            )

            if res.returncode != 0:
                killed += 1
                status = "KILLED"
            else:
                survived += 1
                status = "SURVIVED"

            print(f"[{idx}/{total}] {status}: {rel_file} - {description}")

        finally:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(original_content)

    score = (killed / total) * 100.0 if total > 0 else 0.0
    print(f"\nMutation Score: {killed}/{total} ({score:.1f}%) killed.")

    if survived > 0:
        print(f"FAILED: {survived} mutant(s) survived.")
        return 1

    print("SUCCESS: All mutants killed by test suite.")
    return 0


if __name__ == "__main__":
    sys.exit(run_mutation_test())
