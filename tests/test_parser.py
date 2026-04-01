from pathlib import Path
import unittest

from nlp_pipeline import PatientParser


BASE_DIR = Path(__file__).resolve().parents[1]


class PatientParserTests(unittest.TestCase):
    def test_sample_chart_extracts_clinical_profile(self):
        text = (BASE_DIR / "Sample_Patient_Chart.txt").read_text(encoding="utf-8", errors="ignore")
        profile = PatientParser().parse(text)

        self.assertEqual(profile["age"], 62)
        self.assertEqual(profile["sex"], "Female")
        self.assertIn("Breast Cancer", profile["diagnoses"])
        self.assertIn("HER2", profile["biomarkers"])
        self.assertTrue(profile["metastatic"])
        self.assertIn("Trastuzumab", profile["therapies"])
        self.assertEqual(profile["performance_status"], "ECOG 1")


if __name__ == "__main__":
    unittest.main()
