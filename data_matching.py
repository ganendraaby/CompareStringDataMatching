# ------------------------------------------------------------
# 1. Modul utilitas Arsitektur Data Matching
# ------------------------------------------------------------
import pandas as pd, numpy as np, recordlinkage as rl
from recordlinkage.preprocessing import clean
from rapidfuzz import fuzz as rf_fuzz
from nltk.metrics.distance import jaro_winkler_similarity
from pathlib import Path

class DataMatchingPipeline:
    """Arsitektur sederhana: Load ➜ Clean ➜ Index ➜ Compare ➜ Classify ➜ Export"""

    def __init__(self, csv_path: str, out_dir: str = "output_dm"):
        self.csv_path = Path(csv_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)
        self.df = pd.read_csv(self.csv_path)

    # --- 1) CLEAN -----------------------------------------------------------
    def clean_data(self):
        cols_needed = ["Authors", "Title", "Source"]
        self.df = self.df.dropna(subset=cols_needed).copy()
        for col in cols_needed:
            self.df[f"{col}_clean"] = clean(
                self.df[col], lowercase=True, remove_brackets=True, strip_accents="unicode"
            )
        # Simpan versi “bersih” bila butuh
        self.df.to_csv(self.out_dir / "clean_popcite.csv", index=False)
        print(f"[CLEAN] DataFrame bersih disimpan → clean_popcite.csv  ({self.df.shape[0]} baris)")

    # --- 2) INDEX -----------------------------------------------------------
    def create_candidate_pairs(self):
        indexer = rl.Index()
        indexer.full()  # semua kombinasi; ganti blocking bila dataset besar
        self.candidate_links = indexer.index(self.df, self.df)
        # buang self-match (idx kiri = idx kanan)
        self.candidate_links = self.candidate_links[
            self.candidate_links.get_level_values(0) != self.candidate_links.get_level_values(1)
        ]
        print(f"[INDEX] Kandidat pasangan: {len(self.candidate_links):,}")

    # --- 3) COMPARE ---------------------------------------------------------
    def compare_pairs(self):
        cmp = rl.Compare()

        # Cosine similarity (RecordLinkage) utk teks normalisasi
        cmp.string("Authors_clean", "Authors_clean", method="cosine", threshold=0.5, label="authors")
        cmp.string("Title_clean", "Title_clean", method="cosine", threshold=0.5, label="title")
        cmp.exact("Source_clean", "Source_clean", label="source_exact")

        # Hitung semua fitur
        self.features = cmp.compute(self.candidate_links, self.df, self.df)
        self.features.to_csv(self.out_dir / "all_pairwise_comparisons.csv")
        print(f"[COMPARE] Fitur per pasangan disimpan → all_pairwise_comparisons.csv")

    # --- 4) CLASSIFY (ECM) --------------------------------------------------
    def classify_pairs(self):
        X = self.features.astype(int)
        model = rl.ECMClassifier()
        model.fit(X)

        # Prediksi & probabilitas
        self.pred_links = model.predict(X)
        self.probabilities = model.prob(X)

        # Buat DataFrame hasil lengkap
        results = []
        for idx, prob in zip(X.index, self.probabilities):
            i1, i2 = idx
            results.append(
                {
                    "Record1_Index": i1,
                    "Record2_Index": i2,
                    "Record1_Authors": self.df.loc[i1, "Authors"],
                    "Record1_Title": self.df.loc[i1, "Title"],
                    "Record1_Source": self.df.loc[i1, "Source"],
                    "Record2_Authors": self.df.loc[i2, "Authors"],
                    "Record2_Title": self.df.loc[i2, "Title"],
                    "Record2_Source": self.df.loc[i2, "Source"],
                    "Match_Probability": prob,
                    "ECM_Prediction": idx in self.pred_links,
                    "Authors_Similarity": X.loc[idx, "authors"],
                    "Title_Similarity": X.loc[idx, "title"],
                    "Source_Exact": X.loc[idx, "source_exact"],
                }
            )
        self.results_df = pd.DataFrame(results).sort_values("Match_Probability", ascending=False)
        self.results_df.to_csv(self.out_dir / "ecm_all_results.csv", index=False)
        print(f"[CLASSIFY] Hasil lengkap ECM → ecm_all_results.csv")

        # Simpan subset-subset penting
        self.results_df[self.results_df["ECM_Prediction"]].to_csv(
            self.out_dir / "ecm_predicted_matches.csv", index=False
        )
        self.results_df[self.results_df["Match_Probability"] > 0.8].to_csv(
            self.out_dir / "ecm_high_confidence_matches.csv", index=False
        )
        print(
            f"[CLASSIFY] Prediksi match: {len(self.pred_links):,} | High-conf: "
            f"{(self.results_df['Match_Probability']>0.8).sum():,}"
        )

    # --- 5) PIPELINE ORCHESTRATOR ------------------------------------------
    def run(self):
        self.clean_data()
        self.create_candidate_pairs()
        self.compare_pairs()
        self.classify_pairs()
        print("\n=== PIPELINE SELESAI ===")

# ------------------------------------------------------------
# 2. Eksekusi pipeline
# ------------------------------------------------------------
pipeline = DataMatchingPipeline("clean_popcite.csv")
pipeline.run()
