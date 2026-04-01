from pathlib import Path
import tempfile
import unittest

import pandas as pd

from matching import TrialMatcher


BASE_DIR = Path(__file__).resolve().parents[1]


def build_fixture_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "NCT Number": "NCT-HER2-001",
                "Study Title": "HER2 Positive Metastatic Breast Cancer Antibody Drug Conjugate Study",
                "Study URL": "https://example.com/her2",
                "Study Status": "RECRUITING",
                "Brief Summary": "Recruiting interventional study for patients with metastatic HER2 positive breast cancer after trastuzumab or pertuzumab exposure.",
                "Conditions": "Breast Cancer|Metastatic Breast Cancer",
                "Interventions": "DRUG: Trastuzumab Deruxtecan|DRUG: Tucatinib",
                "Locations": "Detroit, Michigan, United States",
                "Study Type": "INTERVENTIONAL",
                "Phases": "PHASE2",
                "Sponsor": "Example Cancer Center",
                "Sex": "FEMALE",
                "Age": "ADULT, OLDER_ADULT",
            },
            {
                "NCT Number": "NCT-LUNG-001",
                "Study Title": "Targeted Therapy Trial in Advanced Lung Cancer",
                "Study URL": "https://example.com/lung",
                "Study Status": "RECRUITING",
                "Brief Summary": "Interventional study for EGFR mutated lung cancer.",
                "Conditions": "Lung Cancer",
                "Interventions": "DRUG: Osimertinib",
                "Locations": "Boston, Massachusetts, United States",
                "Study Type": "INTERVENTIONAL",
                "Phases": "PHASE2",
                "Sponsor": "Example Lung Institute",
                "Sex": "ALL",
                "Age": "ADULT, OLDER_ADULT",
            },
            {
                "NCT Number": "NCT-OBS-001",
                "Study Title": "Observational Breast Cancer Registry",
                "Study URL": "https://example.com/registry",
                "Study Status": "COMPLETED",
                "Brief Summary": "Completed observational registry for breast cancer outcomes.",
                "Conditions": "Breast Cancer",
                "Interventions": "",
                "Locations": "Chicago, Illinois, United States",
                "Study Type": "OBSERVATIONAL",
                "Phases": "",
                "Sponsor": "Registry Group",
                "Sex": "ALL",
                "Age": "ADULT, OLDER_ADULT",
            },
        ]
    )


class TrialMatcherTests(unittest.TestCase):
    def test_breast_cancer_patient_prefers_her2_trial(self):
        sample_text = (BASE_DIR / "Sample_Patient_Chart.txt").read_text(encoding="utf-8", errors="ignore")
        with tempfile.TemporaryDirectory() as tmpdir:
            matcher = TrialMatcher(
                build_fixture_dataframe(),
                cache_dir=tmpdir,
                use_gpu=False,
                enable_semantic=False,
            )
            response = matcher.search(
                sample_text,
                top_k=2,
                location="Detroit, Michigan",
                mode="balanced",
                active_only=True,
                interventional_only=True,
            )

        self.assertTrue(response["results"])
        top_result = response["results"][0]
        self.assertEqual(top_result["nct_number"], "NCT-HER2-001")
        self.assertEqual(top_result["status"], "Recruiting")
        self.assertIn("Condition overlap", " ".join(top_result["reasons"]))


if __name__ == "__main__":
    unittest.main()
