"""
Unit tests for the simplified Lectrace presentation (`Lectrace/`) and Showcase Landing Page (`docs/index.html`).

Verifies:
1. `Lectrace/` contains ONLY `_project_data.py` and `01_gbm_ood_presentation.py`.
2. `Lectrace/_project_data.py` provides reproducible GBM cell-line, drug, and benchmark inputs
   implementing the `__lectrace__()` serialization protocol.
3. `Lectrace/01_gbm_ood_presentation.py` defines `main()`, isolates core `lectrace` imports
   from optional `table` import, uses official Lectrace directives and rendering calls, and
   deterministically executes all 4 selected execution states.
4. `docs/index.html`, `lectrace.toml`, `README.md`, `pyproject.toml`, and `.github/workflows/ci.yml`
   configure and document the single-file showcase landing page in `/docs`, official Lectrace PyPI package
   (`pip install lectrace`), CLI commands, and Python 3.14 environment in GitHub Actions.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LECTRACE_DIR = REPO_ROOT / "Lectrace"
if str(LECTRACE_DIR) not in sys.path:
    sys.path.insert(0, str(LECTRACE_DIR))

import _project_data


def _load_presentation_module():
    script_path = LECTRACE_DIR / "01_gbm_ood_presentation.py"
    spec = importlib.util.spec_from_file_location("gbm_lectrace_pres", str(script_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestLectraceDirectoryAndProjectData(unittest.TestCase):
    """Verify the 2-file Lectrace directory layout and `_project_data.py` inputs."""

    def test_lectrace_directory_contains_only_two_files(self):
        files = {
            p.name
            for p in LECTRACE_DIR.iterdir()
            if p.name != "__pycache__" and not p.name.startswith(".")
        }
        self.assertEqual(files, {"_project_data.py", "01_gbm_ood_presentation.py"})

    def test_cohort_summary_constants(self):
        summary = _project_data.COHORT_SUMMARY
        self.assertEqual(summary["deduplicated_pairs"], 5107)
        self.assertEqual(summary["unique_cell_lines"], 24)
        self.assertEqual(summary["unique_drugs"], 243)
        self.assertEqual(summary["morgan_fp_bits"], 2048)

    def test_sample_cell_lines_and_lectrace_protocol(self):
        cells = _project_data.SAMPLE_CELL_LINES
        self.assertEqual(len(cells), 4)
        u87, t98g = cells[0], cells[1]
        self.assertEqual(u87.cell_line_name, "U-87 MG")
        self.assertEqual(t98g.cell_line_name, "T98G")
        self.assertLess(u87.mgmt_expr, t98g.mgmt_expr)

        serialized = u87.__lectrace__()
        self.assertIsInstance(serialized, dict)
        self.assertIn("MGMT_Expr", serialized)
        self.assertEqual(len(serialized["pca_4d"]), 4)

    def test_sample_drugs_and_lectrace_protocol(self):
        drugs = _project_data.SAMPLE_DRUGS
        self.assertEqual(len(drugs), 4)
        tmz = drugs[0]
        self.assertEqual(tmz.drug_name, "Temozolomide")
        self.assertEqual(tmz.split_assignment, "test_scaffold")

        serialized = tmz.__lectrace__()
        self.assertEqual(serialized["drug_name"], "Temozolomide")
        self.assertAlmostEqual(serialized["true_LN_IC50"], 6.45, places=2)
        self.assertAlmostEqual(serialized["pred_scaffold_LN_IC50"], 3.32, places=2)


class TestLectracePresentationStates(unittest.TestCase):
    """Test the 4 selected execution states in `01_gbm_ood_presentation.py`."""

    @classmethod
    def setUpClass(cls):
        cls.pres = _load_presentation_module()

    def test_script_directives_and_isolated_imports_present(self):
        script_text = (LECTRACE_DIR / "01_gbm_ood_presentation.py").read_text(encoding="utf-8")
        self.assertIn("def main():", script_text)
        self.assertIn("from lectrace import Reference, link, note, plot, text", script_text)
        self.assertIn("from lectrace import table", script_text)
        self.assertIn("# @inspect", script_text)
        self.assertIn("# @clear", script_text)
        self.assertIn("# @stepover", script_text)
        for fn_call in ("text(", "note(", "table(", "plot(", "link(", "Reference("):
            self.assertIn(fn_call, script_text)

    def test_state_1_leakage_free_preprocessing(self):
        res1 = self.pres.trace_state_1_leakage_free_preprocessing(
            _project_data.SAMPLE_CELL_LINES, _project_data.SAMPLE_DRUGS
        )
        self.assertTrue(res1["leakage_audit_passed"])
        self.assertEqual(res1["train_cells"], ["U-87 MG", "T98G"])
        self.assertEqual(res1["heldout_test_cells"], ["GAMG", "LN-229"])
        self.assertIn("Temozolomide", res1["heldout_test_drugs"])

    def test_state_2_two_tower_forward_pass(self):
        res2 = self.pres.trace_state_2_two_tower_forward_pass(
            _project_data.SAMPLE_DRUGS[0],
            _project_data.SAMPLE_CELL_LINES[0],
            _project_data.SAMPLE_CELL_LINES[1],
            _project_data.DEMO_MLP_WEIGHTS,
        )
        self.assertEqual(res2["drug"], "Temozolomide")
        self.assertEqual(len(res2["drug_emb_4d"]), 4)
        self.assertGreater(res2["t98g_mgmt_high_pred_ic50"], res2["u87_mgmt_low_pred_ic50"])
        self.assertGreater(res2["mgmt_resistance_shift_ln_ic50"], 0.0)

    def test_state_3_shortcut_decomposition(self):
        res3 = self.pres.trace_state_3_shortcut_decomposition(_project_data.BENCHMARK_RESULTS)
        self.assertAlmostEqual(res3["additive_baseline_spearman"], 0.8752, places=4)
        self.assertAlmostEqual(res3["mlp_global_spearman"], 0.8744, places=4)
        self.assertAlmostEqual(res3["interaction_gain_global"], -0.0008, places=4)
        self.assertAlmostEqual(res3["interaction_gain_within_cell"], -0.0013, places=4)

    def test_state_4_ood_shrinkage_and_decoupling(self):
        res4 = self.pres.trace_state_4_ood_shrinkage_and_decoupling(
            _project_data.BENCHMARK_RESULTS, _project_data.SAMPLE_DRUGS
        )
        self.assertLess(
            res4["global_spearman"]["scaffold_chemical_ood"],
            res4["global_spearman"]["random"],
        )
        self.assertAlmostEqual(
            res4["shrinkage_ratio_pred_over_true_std"]["scaffold"], 0.663, places=3
        )
        self.assertGreaterEqual(res4["within_drug_median_spearman"]["scaffold"], 0.50)
        self.assertAlmostEqual(res4["tmz_scaffold_bias_ln_ic50"], -3.13, places=2)
        self.assertAlmostEqual(res4["irinotecan_scaffold_bias_ln_ic50"], -3.78, places=2)

    def test_main_runs_cleanly(self):
        self.pres.main()


class TestLectraceReadmeShowcaseAndCI(unittest.TestCase):
    """Verify docs/index.html, lectrace.toml, README.md, and CI workflow configuration."""

    def test_index_html_showcase_exists_and_valid(self):
        docs_html_path = REPO_ROOT / "docs" / "index.html"
        root_html_path = REPO_ROOT / "index.html"
        html_path = docs_html_path if docs_html_path.is_file() else root_html_path
        self.assertTrue(
            html_path.is_file(),
            "Expected showcase index.html in docs/index.html (or root index.html)",
        )
        html_text = html_path.read_text(encoding="utf-8")
        self.assertIn("<!DOCTYPE html>", html_text)
        self.assertIn("./lectrace/", html_text)
        self.assertIn("0.8744", html_text)
        self.assertIn("0.8752", html_text)
        self.assertIn("0.2905", html_text)
        self.assertIn("0.530", html_text)
        self.assertIn("Random Split Interaction Gain", html_text)
        self.assertIn("One model, three views of what it memorized", html_text)
        self.assertIn("0.87 Spearman was a drug-mean illusion", html_text)
        self.assertIn("Relative ranking yes, absolute dosing no", html_text)
        self.assertNotIn("<input", html_text)
        self.assertNotIn("\\sigma", html_text)
        self.assertNotIn("$0.530$", html_text)

    def test_lectrace_toml_config_exists_and_valid(self):
        toml_path = REPO_ROOT / "lectrace.toml"
        self.assertTrue(toml_path.is_file())
        content = toml_path.read_text(encoding="utf-8")
        self.assertIn("[lectrace]", content)
        self.assertIn("Lectrace/01_gbm_ood_presentation.py", content)
        self.assertIn('output = "docs/lectrace"', content)

    def test_readme_documents_lectrace_cli_and_states(self):
        readme_path = REPO_ROOT / "README.md"
        self.assertTrue(readme_path.is_file())
        content = readme_path.read_text(encoding="utf-8")
        self.assertIn("pip install lectrace", content)
        self.assertIn("lectrace serve Lectrace/01_gbm_ood_presentation.py", content)
        self.assertIn("docs/index.html", content)
        self.assertNotIn("git+https://github.com/CSC3160/lectrace.git", content)
        self.assertNotIn("--workspace-root", content)
        for state_idx in range(1, 5):
            self.assertIn(f"Execution State {state_idx}", content)

    def test_ci_workflow_deploys_showcase_and_lectrace_on_python_314(self):
        ci_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
        self.assertTrue(ci_path.is_file())
        ci_text = ci_path.read_text(encoding="utf-8")
        self.assertIn("pip install lectrace", ci_text)
        self.assertIn("docs/index.html", ci_text)
        self.assertIn(
            "lectrace build Lectrace/01_gbm_ood_presentation.py --output docs/lectrace --no-incremental",
            ci_text,
        )
        self.assertIn("touch docs/.nojekyll", ci_text)
        self.assertTrue((REPO_ROOT / "docs" / "index.html").is_file())
        self.assertTrue((REPO_ROOT / "docs" / ".nojekyll").is_file())
        self.assertIn('"3.14"', ci_text)
        self.assertNotIn('"3.10"', ci_text)
        self.assertNotIn('"3.11"', ci_text)
        self.assertNotIn("git+https://github.com/CSC3160/lectrace.git", ci_text)


if __name__ == "__main__":
    unittest.main()
