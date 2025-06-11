import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.sparse import hstack

# === Configurable ===
input_path = "traced_partial.csv"
output_path = "sampled_final.csv"
n_samples = 1000  # Adjust to 1000 later
random_state = 42
image_weight = 0.25  # Scale down image vector influence
# ====================

# Step 1: Load and filter correct answers
df = pd.read_csv(input_path)
print(f"size of original input: {len(df)}")
df_filtered = df[df["generated_answer"].str.strip() == df["Answer_label"].str.strip()].copy()
print(f"size of filtered: {len(df_filtered)}")
# Step 2a: Vectorize questions
question_vectorizer = TfidfVectorizer(max_features=512)
question_vecs = question_vectorizer.fit_transform(df_filtered["question"].astype(str))

# Step 2b: Vectorize image IDs (just file stem or full path)
image_vectorizer = TfidfVectorizer()
image_vecs = image_vectorizer.fit_transform(df_filtered["figure_path"].astype(str))

# Step 2c: Apply weight to image vector and combine
image_vecs = image_vecs.multiply(image_weight)  # Scale down image influence
X = hstack([question_vecs, image_vecs])
X = normalize(X)  # Optional: L2 normalize before clustering

# Step 3: KMeans clustering
k = min(n_samples, len(df_filtered))
kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
clusters = kmeans.fit_predict(X)

# Step 4: Sample one item per cluster
df_filtered["cluster"] = clusters
sampled_rows = []

for c in range(k):
    cluster_df = df_filtered[df_filtered["cluster"] == c]
    if len(cluster_df) == 0:
        continue
    sampled_rows.append(cluster_df.sample(n=1, random_state=random_state))

# Step 5: Save result
df_sampled = pd.concat(sampled_rows).drop(columns=["cluster"])
df_sampled.to_csv(output_path, index=False)

print(f" Sampled {len(df_sampled)} diverse rows to {output_path}")