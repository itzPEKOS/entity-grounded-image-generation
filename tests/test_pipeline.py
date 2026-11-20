"""End-to-end unit and integration tests for the multi-agent pipeline."""

import tempfile
import unittest
from pathlib import Path
from src.orchestrator import EntityGroundedOrchestrator


class TestEntityGroundedPipeline(unittest.TestCase):
    def test_end_to_end_demo_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            orchestrator = EntityGroundedOrchestrator(max_retries=2, output_root=Path(tmp_dir))
            prompt = "A Shiba Inu sitting in front of Wat Arun at golden hour"
            result = orchestrator.run(prompt, run_id="test_run")

            entity_ids = {e.entity_id for e in result.extraction.entities}
            self.assertIn("shiba_inu", entity_ids)
            self.assertIn("wat_arun", entity_ids)

            self.assertEqual(len(result.constraint_spec.entity_specs), 2)
            self.assertGreaterEqual(len(result.constraint_spec.all_checks()), 6)

            self.assertEqual(len(result.reports), 2)
            self.assertFalse(result.reports[0].overall_passed)
            self.assertTrue(result.final_report.overall_passed)
            self.assertEqual(result.final_report.pass_rate, 1.0)

            self.assertTrue(result.final_image_path.exists())
            self.assertTrue((result.run_dir / "02_constraint_spec.json").exists())
            self.assertTrue((result.run_dir / "verification_report_attempt_2.md").exists())


if __name__ == "__main__":
    unittest.main()
