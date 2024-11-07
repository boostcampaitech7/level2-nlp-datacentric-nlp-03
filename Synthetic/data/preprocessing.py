import pandas as pd
import csv

# 데이터 불러오기
df1 = pd.read_csv("/data/ephemeral/home/level2-nlp-datacentric-nlp-03/Synthetic/data/generated_news_titles_0.csv", escapechar='\\')
df2 = pd.read_csv("/data/ephemeral/home/level2-nlp-datacentric-nlp-03/Synthetic/data/generated_news_titles_1.csv", escapechar='\\')
df3 = pd.read_csv("/data/ephemeral/home/level2-nlp-datacentric-nlp-03/Synthetic/data/generated_news_titles_2.csv", escapechar='\\')
df4 = pd.read_csv("/data/ephemeral/home/level2-nlp-datacentric-nlp-03/Synthetic/data/generated_news_titles_3.csv", escapechar='\\')

combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)
# text 열에서 앞뒤 따옴표 제거
print(repr(df3['text'][0]))

# 전처리 과정
# 1. 맨 앞과 맨 뒤의 따옴표 제거
# combined_df['text'] = combined_df['text'].str.replace('"', '', regex=False).str.replace('\\,', ',', regex=False)

# 2. "IT 기업들"이 포함된 문장 제거
combined_df = combined_df[~combined_df['text'].str.contains("IT 기업들")]

shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)

combined_df['ID'] = [f"ynat-v1_train_{str(i).zfill(5)}" for i in range(1, len(combined_df) + 1)]

# 컬럼 순서 변경 (ID, text, target 순으로)
combined_df = combined_df[['ID', 'text', 'target']]

shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)

# 셔플된 데이터프레임을 CSV로 저장
shuffled_df.to_csv('shuffled_data.csv', index=False)
