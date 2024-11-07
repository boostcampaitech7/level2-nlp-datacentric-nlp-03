import pandas as pd
import csv

# 데이터 불러오기
df1 = pd.read_csv("/data/ephemeral/home/level2-nlp-datacentric-nlp-03/Synthetic/data/generated_news_titles_0.csv")
df2 = pd.read_csv("/data/ephemeral/home/level2-nlp-datacentric-nlp-03/Synthetic/data/generated_news_titles_1.csv")
df3 = pd.read_csv("/data/ephemeral/home/level2-nlp-datacentric-nlp-03/Synthetic/data/generated_news_titles_2.csv")
df4 = pd.read_csv("/data/ephemeral/home/level2-nlp-datacentric-nlp-03/Synthetic/data/generated_news_titles_3.csv")

# 데이터프레임 합치기
combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# 전처리 과정
# 1. text 열에서 모든 종류의 따옴표 제거
quotation_pattern = r'[\"“”‘’«»„\\,]'
combined_df['text'] = combined_df['text'].str.replace(quotation_pattern, '', regex=True)

# 2. "IT 기업들"이 포함된 문장 제거 (예시로 준 문장을 카피하는 경우가 있었음)
combined_df = combined_df[~combined_df['text'].str.contains("IT 기업들")]

# 3. ID 생성
combined_df['ID'] = [f"ynat-v1_train_{str(i).zfill(5)}" for i in range(1, len(combined_df) + 1)]

# 4. 컬럼 순서 변경
combined_df = combined_df[['ID', 'text', 'target']]

# 5. 데이터프레임 셔플
shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)

# 셔플된 데이터프레임을 CSV로 저장
shuffled_df.to_csv('shuffled_data.csv', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
