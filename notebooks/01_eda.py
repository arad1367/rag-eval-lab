# %% [markdown]
# # Exploratory Data Analysis (EDA) of the arXiv abstracts
# In VS Code, click "Run Cell" above each `# %%` line (or press Shift+Enter).
# Goal: understand the data BEFORE building anything. How long are the abstracts?
# Any duplicates or empty rows? This decides our chunk sizes.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from rag_lab.config import PAPERS_PATH

df = pd.read_json(PAPERS_PATH, lines=True)
print(df.shape)  # (rows, columns)
df.head()

# %% Data quality checks
print("Missing values per column:\n", df.isna().sum())
print("Duplicate ids:", df["id"].duplicated().sum())
print("Empty abstracts:", (df["abstract"].str.strip() == "").sum())

# %% How long are the abstracts (in words)?
df["n_words"] = df["abstract"].str.split().str.len()
print(df["n_words"].describe())  # count, mean, std, min, quartiles, max

# %%
df["n_words"].plot(kind="hist", bins=30, title="Abstract length (words)")
plt.xlabel("words")
plt.show()

# %% Which words appear most often in titles? (a quick look at the topics)
words = df["title"].str.lower().str.findall(r"[a-z]{4,}").explode()
words.value_counts().head(20)

# %% [markdown]
# **What to write down (you will use it in the README and in interviews):**
# - Median abstract length is about N words, so a 300-word chunk = usually the whole
#   abstract, and a 50-word chunk = about 3-5 chunks per paper.
# - Data quality issues you found (duplicates, empty rows) and how you handled them.
# %%
