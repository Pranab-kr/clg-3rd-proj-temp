import subprocess
import sys
from pathlib import Path


def test_generate_data_script_writes_all_tables(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "data"

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "generate_data.py"),
            "--seed",
            "42",
            "--output-dir",
            str(output_dir),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Validation: passed" in result.stdout
    assert {path.name for path in output_dir.glob("*.csv")} == {
        "products.csv",
        "users.csv",
        "interactions.csv",
    }
