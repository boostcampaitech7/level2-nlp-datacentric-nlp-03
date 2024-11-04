import pandas as pd 

df = pd.read_csv("/data/ephemeral/home/Yunseo_DCTC/code/last.csv")

df_filtered = df[~df['text'].str.contains('hate-speech|₩', regex=True, na=False)]


df_filtered.to_csv("last.csv", index=False)