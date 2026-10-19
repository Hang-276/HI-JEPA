"""Launcher behavior: portable subsets, frozen budgets, and fail-closed gates.

These tests generate manifests or stub subprocesses; they launch no experiment.
"""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from hijepa.config import expand_matrix
from hijepa.io import protocol_fingerprint

spec = importlib.util.spec_from_file_location("hijepa_experiment_launcher", REPO / "scripts" / "_experiment.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class ScriptTests(unittest.TestCase):
    def shell(self, name, profile, output, cwd):
        environment = dict(os.environ, HIJEPA_PYTHON=sys.executable)
        result = subprocess.run(["bash", str(REPO / "scripts" / name), profile,
                                 str(output), "--manifest-only"], cwd=cwd,
                                env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads((Path(cwd) / output / "planned_manifest.json").read_text())

    def test_all_experiment_shell_manifests_match_canonical_budgets(self):
        expected = {"formal": [39, 12, 12, 57], "local": [12, 8, 6, 24]}
        with tempfile.TemporaryDirectory(prefix="hijepa scripts ") as directory:
            for profile, counts in expected.items():
                canonical = {cfg["name"]: cfg for cfg in expand_matrix(REPO / "configs" / f"matrix_{profile}.json")}
                names = ["run_experiment1.sh", "run_experiment2.sh", "run_experiment3.sh", "run_all.sh"]
                for index, script in enumerate(names):
                    destination = Path(f"results with spaces {profile} {index}")
                    manifest = self.shell(script, profile, destination, directory)
                    self.assertEqual(manifest["run_count"], counts[index])
                    matrix = Path(directory) / destination / "launch" / "matrix.json"
                    self.assertEqual(json.loads(matrix.read_text())["base"], "base.json")
                    for cfg in expand_matrix(matrix):
                        self.assertEqual(protocol_fingerprint(cfg), protocol_fingerprint(canonical[cfg["name"]]))
                    files = [p.name for p in (Path(directory)/destination).rglob("*") if p.is_file()]
                    self.assertNotIn("doctor_cuda.json", files)
                    self.assertNotIn("checkpoint.pt", files)
                    self.assertNotIn("pseudo_solution_audit.json", files)

    def test_reuse_dependencies_are_present_and_ordered(self):
        for profile, source in (("formal", "e1_baseline"), ("local", "e1_real")):
            matrix, selection = launcher.select_matrix("e2", profile)
            self.assertIn(source, selection["dependency_jobs"])
            names = [job["name"] for job in matrix["jobs"]]
            self.assertLess(names.index(source), names.index("e2_full"))
        _, selection = launcher.select_matrix("e3", "formal")
        self.assertIn("e1_baseline", selection["dependency_jobs"])

    def test_formal_acceptance_failure_never_launches_other_tasks(self):
        configs = expand_matrix(REPO / "configs" / "matrix_formal.json")
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            def fake_cli(*arguments, **kwargs):
                calls.append(arguments[0])
                self.assertEqual(arguments[0], "run")
                target = Path(arguments[arguments.index("--output")+1])
                target.mkdir(parents=True, exist_ok=True)
                launcher.write_json(target/"result.json", {"status": "completed", "anticollapse": {"passed": False}})
            with mock.patch.object(launcher, "cli", side_effect=fake_cli), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    launcher.formal_acceptance(configs, output)
            self.assertEqual(calls, ["run", "run"])
            self.assertFalse(json.loads((output/"prelaunch_anticollapse_acceptance.json").read_text())["passed"])

    def test_frozen_launch_rejects_changed_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"selection.json"
            launcher.frozen_json(path, {"experiment": "e1"})
            before = path.read_bytes()
            with self.assertRaises(RuntimeError):
                launcher.frozen_json(path, {"experiment": "e2"})
            self.assertEqual(before, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
