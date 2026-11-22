import subprocess
import sys
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "check_environment.py"
)


class CheckEnvironmentTests(unittest.TestCase):
    def test_reports_wrong_interpreter_and_activation_command(self):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--expected-prefix",
                "/definitely/not/the/climax/environment",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("Python executable:", result.stdout)
        self.assertIn("not the expected Conda environment", result.stderr)
        self.assertIn("conda activate", result.stderr)
        self.assertIn("Environment verification failed", result.stderr)


if __name__ == "__main__":
    unittest.main()
