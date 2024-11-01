from transformers import AutoTokenizer
import os
import re
import pandas as pd

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
OUTPUT_DIR = os.path.join(BASE_DIR, '../output')

df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
# GPT-2 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

# 텍스트 데이터를 토크나이즈하고 다시 디코딩하는 함수
def tokenize_and_decode(text):
    # 텍스트를 토크나이즈
    tokenized = tokenizer.encode(text) 
    # 토큰 ID를 다시 텍스트로 디코딩
    decoded_text = tokenizer.decode(tokenized)
    return decoded_text

# 데이터셋의 텍스트 컬럼에 토크나이즈 및 디코딩 적용
df["decoded_text"] = df["text"].apply(tokenize_and_decode)

df.to_csv(os.path.join(OUTPUT_DIR, 'tokenized_df.csv'), index=False)