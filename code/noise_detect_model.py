from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import re
import pandas as pd
from tqdm import tqdm


BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
OUTPUT_DIR = os.path.join(BASE_DIR, '../output')

df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

df = df[9:11]

# 언어 모델과 토크나이저 불러오기 (예: GPT-2)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# 로그 확률을 계산하여 노이즈 여부를 판단하는 함수
def is_noisy_based_on_log_prob(text, threshold=-5.0):
    # 입력 텍스트를 모델에 맞게 토큰화하고 텐서로 변환
    inputs = tokenizer(text, return_tensors="pt")
    
    # 모델로 로그 확률 계산
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        log_probs = outputs.logits.log_softmax(dim=-1)
        print(log_probs)
    
    # 각 토큰의 로그 확률을 추출하여 평균 계산
    token_log_probs = torch.gather(log_probs, -1, inputs["input_ids"].unsqueeze(-1)).squeeze(-1)
    avg_log_prob = token_log_probs.mean().item()
    
    # 평균 로그 확률이 기준보다 낮으면 노이즈로 판단
    return avg_log_prob

tqdm.pandas()

# 데이터셋에 로그 확률 기반의 노이즈 판별을 적용하면서 진행 상황을 표시
df["language_model_noise"] = df["text"].progress_apply(is_noisy_based_on_log_prob)

df.to_csv(os.path.join(OUTPUT_DIR, 'language_model_noise.csv'), index=False)