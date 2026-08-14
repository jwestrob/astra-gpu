import json
import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_benchmark.py"


class RunBenchmarkTests(unittest.TestCase):
    @staticmethod
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_child_environment_override_is_recorded_without_changing_parent(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            record = root / "record.json"
            stdout = root / "stdout"
            stderr = root / "stderr"
            variable = "PLAN7_RUNNER_TEST_VALUE"
            previous = os.environ.get(variable)
            subprocess.run(
                [
                    str(RUNNER),
                    "--output",
                    str(record),
                    "--stdout",
                    str(stdout),
                    "--stderr",
                    str(stderr),
                    "--label",
                    "environment-test",
                    "--set-env",
                    f"{variable}=child-only",
                    "--",
                    "/usr/bin/env",
                ],
                cwd=PROJECT_ROOT,
                check=True,
            )
            value = json.loads(record.read_text(encoding="utf-8"))
            self.assertEqual(value["environment_overrides"], {variable: "child-only"})
            self.assertIn(f"{variable}=child-only\n", stdout.read_text())
            self.assertEqual(os.environ.get(variable), previous)

    def test_duplicate_environment_override_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            process = subprocess.run(
                [
                    str(RUNNER),
                    "--output",
                    str(root / "record.json"),
                    "--stdout",
                    str(root / "stdout"),
                    "--stderr",
                    str(root / "stderr"),
                    "--label",
                    "duplicate-test",
                    "--set-env",
                    "PLAN7_DUPLICATE=one",
                    "--set-env",
                    "PLAN7_DUPLICATE=two",
                    "--",
                    "/bin/true",
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("only be overridden once", process.stderr)

    def test_child_path_selects_and_prehashes_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable = root / "selected-command"
            executable.write_text(
                "#!/bin/sh\nprintf 'selected\\n'\nprintf '#!/bin/sh\\nexit 7\\n' > \"$0\"\n",
                encoding="utf-8",
            )
            executable.chmod(0o755)
            before = self.sha256(executable)
            record = root / "record.json"
            stdout = root / "stdout"

            subprocess.run(
                [
                    str(RUNNER),
                    "--output",
                    str(record),
                    "--stdout",
                    str(stdout),
                    "--stderr",
                    str(root / "stderr"),
                    "--label",
                    "path-test",
                    "--set-env",
                    f"PATH={root}",
                    "--",
                    executable.name,
                ],
                cwd=PROJECT_ROOT,
                check=True,
            )

            value = json.loads(record.read_text(encoding="utf-8"))
            self.assertEqual(stdout.read_text(encoding="utf-8"), "selected\n")
            self.assertEqual(value["executable"]["path"], str(executable))
            self.assertEqual(value["executable"]["sha256"], before)
            self.assertNotEqual(self.sha256(executable), before)


if __name__ == "__main__":
    unittest.main()
