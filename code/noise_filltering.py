import os
import re
import pandas as pd

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
OUTPUT_DIR = os.path.join(BASE_DIR, '../output')

input_file = os.path.join(DATA_DIR, 'train.csv')
output_file = os.path.join(OUTPUT_DIR, 'noise_isolated.csv')

df = pd.read_csv(os.path.join(DATA_DIR, 'rmkor_filtered.csv'))

# 정규식 패턴 정의
pattern = r'[가-힣]+|[0-9]+|[A-Z]{2,}|[\u4E00-\u9FFF]+'

def clean_text(text):
    if isinstance(text, str):
        matches = re.findall(pattern, text)
        return ''.join(matches)
    return ''

# 'text' 열을 처리하여 필요한 문자만 남김
df['cleaned_text'] = df['text'].apply(clean_text)

# 결과 확인
print(df[['text', 'cleaned_text']])