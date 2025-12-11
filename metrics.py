# metrics.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RESULT_DIR = "predictions_mlp_big"
CSV_PATH = os.path.join(RESULT_DIR, "test_metrics.csv")

# Load test results
df = pd.read_csv(CSV_PATH)

print("==== Test Summary Metrics (Test Set) ====")
print(f"Mean Precision : {df['precision'].mean():.4f} ± {df['precision'].std():.4f}")
print(f"Mean Recall    : {df['recall'].mean():.4f} ± {df['recall'].std():.4f}")
print(f"Mean F1-Score  : {df['f1'].mean():.4f} ± {df['f1'].std():.4f}")

# === Distribution plots ===
plt.figure(figsize=(7, 4))
sns.kdeplot(df['precision'], fill=True, label='Precision', linewidth=2)
sns.kdeplot(df['recall'], fill=True, label='Recall', linewidth=2)
sns.kdeplot(df['f1'], fill=True, label='F1', linewidth=2)
plt.title("Distribution of Per-Image Metrics (Test Set)")
plt.xlabel("Score")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
dist_path = os.path.join(RESULT_DIR, "metric_distributions.png")
plt.savefig(dist_path, dpi=300)
plt.show()

# === Pairplot for relationship between metrics ===
pairplot = sns.pairplot(df[['precision', 'recall', 'f1']], diag_kind='kde', corner=True)
pairplot.fig.suptitle("Pairwise Relationships between Metrics", y=1.02)
pairplot.fig.tight_layout()
pairplot_path = os.path.join(RESULT_DIR, "metric_pairplot.png")
pairplot.fig.savefig(pairplot_path, dpi=300)
plt.show()

print("\nSaved plots:")
print(" -", dist_path)
print(" -", pairplot_path)
