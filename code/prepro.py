import pandas as pd 
import numpy as np

df = pd.read_csv("/data/ephemeral/home/Yunseo_DCTC/data/last.csv")

df_filtered = df[~df['text'].str.contains('hate-speech|<0xE2><0x82><0xA9>', regex=True, na=False) & 
                    ~df['text'].str.fullmatch(r'[A-Za-z\s]+') &                            # 영어로만 이루어진 문장 여부
                    (df['text'].str.len() <= 100)     
                ]

# n = 1
# df_expanded = df_filtered.reindex(np.tile(df_filtered.index, n)).reset_index(drop=True)

df_filtered.to_csv("last.csv", index=False)