# ------------------------------------------------------------
# 1. Instalasi paket (jalankan sekali di notebook / terminal)
# ------------------------------------------------------------
# !pip install pandas ace_tools  # ace_tools otomatis ada di lingkungan ChatGPT

# ------------------------------------------------------------
# 2. Import dan fungsi utilitas
# ------------------------------------------------------------
import pandas as pd
import itertools
import os

def jaccard_authors(s1: str, s2: str) -> float:
    """
    Hitung Jaccard similarity dua string penulis.
    Tokenisasi sederhana: pisah dengan koma, trim spasi, lowercase.
    """
    t1 = {tok.strip().lower() for tok in str(s1).split(',') if tok.strip()}
    t2 = {tok.strip().lower() for tok in str(s2).split(',') if tok.strip()}
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)

# ------------------------------------------------------------
# 3. Baca data
# ------------------------------------------------------------
INPUT_CSV  = "PoPCites.csv"            # ganti jika nama / lokasi berbeda
OUTPUT_CSV = "authors_jaccard_similarity.csv"

df = pd.read_csv(INPUT_CSV)

# Pastikan kolom Authors ada
if "Authors" not in df.columns:
    raise ValueError(f"Kolom 'Authors' tidak ditemukan di {INPUT_CSV}")

# ------------------------------------------------------------
# 4. Hitung Jaccard untuk semua kombinasi unik
# ------------------------------------------------------------
pairs = []
for (i, authors_i), (j, authors_j) in itertools.combinations(
        df["Authors"].dropna().items(), 2
    ):
    score = jaccard_authors(authors_i, authors_j)
    pairs.append(
        {
            "Index1": i,
            "Index2": j,
            "Authors1": authors_i,
            "Authors2": authors_j,
            "Jaccard_Similarity": round(score, 3),
        }
    )

pairs_df = pd.DataFrame(pairs)
print(f"Total pasangan dihitung: {len(pairs_df):,}")

# ------------------------------------------------------------
# 5. Simpan hasil penuh ke CSV
# ------------------------------------------------------------
pairs_df.to_csv(OUTPUT_CSV, index=False)
print(f"Hasil lengkap tersimpan -> {os.path.abspath(OUTPUT_CSV)}")

# ------------------------------------------------------------
# 6. Tampilkan 25 skor tertinggi (opsional)
# ------------------------------------------------------------
top_df = pairs_df.sort_values("Jaccard_Similarity", ascending=False).head(25)