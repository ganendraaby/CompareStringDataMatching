import pandas as pd
import rltk
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import re

class PopCiteRecord(rltk.Record):
    @property
    def id(self):
        return str(self.raw_object.get('id', '')).strip()
    
    @property
    def authors(self):
        return str(self.raw_object.get('Authors', '')).strip()
    
    @property
    def title(self):
        return str(self.raw_object.get('Title', '')).strip()
    
    @property
    def source(self):
        return str(self.raw_object.get('Source', '')).strip()
    
    def get_title_tokens(self):
        return re.findall(r'\b\w+\b', self.title.lower())
    
    def get_authors_tokens(self):
        return re.findall(r'\b\w+\b', self.authors.lower())
    
    def get_source_tokens(self):
        return re.findall(r'\b\w+\b', self.source.lower())

def jaccard_similarity(set1, set2):
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def compute_similarity_scores(r1, r2):
    scores = {}
    scores['title_levenshtein'] = rltk.levenshtein_similarity(r1.title, r2.title)
    try:
        scores['title_jaro_winkler'] = rltk.jaro_winkler_similarity(r1.title, r2.title)
    except AttributeError:
        scores['title_jaro_winkler'] = scores['title_levenshtein']
    title_tokens1 = set(r1.get_title_tokens())
    title_tokens2 = set(r2.get_title_tokens())
    scores['title_jaccard'] = jaccard_similarity(title_tokens1, title_tokens2)
    scores['authors_levenshtein'] = rltk.levenshtein_similarity(r1.authors, r2.authors)
    authors_tokens1 = set(r1.get_authors_tokens())
    authors_tokens2 = set(r2.get_authors_tokens())
    scores['authors_jaccard'] = jaccard_similarity(authors_tokens1, authors_tokens2)
    source_tokens1 = set(r1.get_source_tokens())
    source_tokens2 = set(r2.get_source_tokens())
    scores['source_jaccard'] = jaccard_similarity(source_tokens1, source_tokens2)
    scores['source_exact'] = 1.0 if r1.source.lower() == r2.source.lower() else 0.0
    try:
        if r1.title.strip() and r2.title.strip():
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([r1.title, r2.title])
            scores['cosine_tfidf'] = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        else:
            scores['cosine_tfidf'] = 0.0
    except:
        scores['cosine_tfidf'] = 0.0
    scores['composite_score'] = (
        0.3 * scores['title_jaro_winkler'] +
        0.2 * scores['title_jaccard'] +
        0.25 * scores['authors_jaccard'] +
        0.15 * scores['source_jaccard'] +
        0.1 * scores['cosine_tfidf']
    )
    return scores

class SimpleRLTKPipeline:
    def __init__(self, csv_path: str, out_dir: str = "output_rltk"):
        self.csv_path = Path(csv_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)
        self.prepared_csv = self.out_dir / "prepared_data.csv"
    
    def prepare_dataset(self):
        print("[PREP] Preparing dataset...")
        df = pd.read_csv(self.csv_path)
        df = df.dropna(subset=["Authors", "Title", "Source"]).copy()
        df = df.reset_index(drop=True)
        if 'id' not in df.columns:
            df.insert(0, "id", df.index.astype(str))
        else:
            df['id'] = df['id'].astype(str)
        df.to_csv(self.prepared_csv, index=False)
        print(f"[PREP] Dataset prepared at: {self.prepared_csv}")
        print(f"[PREP] Total records: {len(df)}")
        return df
    
    def load_rltk_dataset(self):
        print("[LOAD] Loading dataset with RLTK...")
        reader = rltk.CSVReader(open(self.prepared_csv, 'r', encoding='utf-8'))
        dataset = rltk.Dataset(reader, record_class=PopCiteRecord)
        print("[LOAD] Dataset loaded")
        return dataset
    
    def generate_all_pairs(self, dataset):
        print("[PAIR] Generating all unique record pairs...")
        records = list(dataset)
        n = len(records)
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                pairs.append((records[i], records[j]))
        print(f"[PAIR] Generated {len(pairs)} pairs")
        return pairs
    
    def compute_all_similarities(self, pairs):
        print("[SIM] Computing similarity scores...")
        results = []
        for idx, (r1, r2) in enumerate(pairs):
            if idx % 100 == 0:
                print(f"[SIM] Processed {idx}/{len(pairs)} pairs")
            try:
                scores = compute_similarity_scores(r1, r2)
                result = {
                    'Record1_Index': r1.id,
                    'Record2_Index': r2.id,
                    'Record1_Authors': r1.authors,
                    'Record1_Title': r1.title,
                    'Record1_Source': r1.source,
                    'Record2_Authors': r2.authors,
                    'Record2_Title': r2.title,
                    'Record2_Source': r2.source,
                    'Match_Probability': round(scores['composite_score'], 8),
                    'ECM_Prediction': scores['composite_score'] > 0.5,
                    'Authors_Similarity': round(scores['authors_jaccard'], 4),
                    'Title_Similarity': round(scores['title_jaccard'], 4),
                    'Source_Exact_Match': round(scores['source_exact'], 4),
                }
                results.append(result)
            except Exception as e:
                print(f"[ERROR] Processing pair {r1.id}-{r2.id}: {e}")
        print(f"[SIM] Computed similarities for {len(results)} pairs")
        return results
    
    def save_results(self, results):
        print("[SAVE] Saving results...")
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('Match_Probability', ascending=False)
        csv_out = self.out_dir / "matching_result_popcites_all.csv"
        df_results.to_csv(csv_out, index=False, sep='\t')
        top_matches = df_results[df_results['Match_Probability'] > 0.5]
        top_csv = self.out_dir / "top_matches.csv"
        top_matches.to_csv(top_csv, index=False, sep='\t')
        summary = {
            'total_pairs': len(df_results),
            'high_similarity_pairs': len(df_results[df_results['Match_Probability'] > 0.7]),
            'medium_similarity_pairs': len(df_results[df_results['Match_Probability'].between(0.4, 0.7)]),
            'low_similarity_pairs': len(df_results[df_results['Match_Probability'] < 0.4]),
            'average_match_probability': float(df_results['Match_Probability'].mean()),
            'max_match_probability': float(df_results['Match_Probability'].max()),
            'min_match_probability': float(df_results['Match_Probability'].min())
        }
        summary_file = self.out_dir / "matching_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[SAVE] Saved full results: {csv_out}")
        print(f"[SAVE] Saved top matches (>0.5): {top_csv}")
        print(f"[SAVE] Saved summary: {summary_file}")
        return df_results, summary
    
    def run(self):
        print("="*50)
        print("Running full pairwise RLTK pipeline")
        print("="*50)
        df = self.prepare_dataset()
        dataset = self.load_rltk_dataset()
        pairs = self.generate_all_pairs(dataset)
        results = self.compute_all_similarities(pairs)
        df_results, summary = self.save_results(results)
        print("\nSummary:")
        print(summary)
        return df_results, summary

if __name__ == "__main__":
    pipeline = SimpleRLTKPipeline("clean_popcite.csv")
    results_df, summary = pipeline.run()
    print("\nTop 10 matches:")
    print(results_df.head(10)[['Record1_Index', 'Record2_Index', 'Record1_Title', 'Record2_Title', 'Match_Probability']])
