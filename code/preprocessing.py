import os
import re
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

login("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")

# Set random seeds for reproducibility
SEED = 456
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Set device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Define directories
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
OUTPUT_DIR = os.path.join(BASE_DIR, '../output')

# Load the Meta-LLaMA 3 8B Instruction model and tokenizer
model_name = 'Bllossom/llama-3.2-Korean-Bllossom-3B'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)

# Load your data
data = pd.read_csv(os.path.join(DATA_DIR, 'clean_df.csv'))


def extract_corrected_text(output):
    # 모든 "corrected_text" 항목을 찾아 리스트로 저장
    matches = re.findall(r'"corrected_text":\s*"([^"]+)"', output)
    # 마지막 항목을 반환
    return matches[-1] if matches else None


# Function to clean a single sentence using the model
def clean_sentence(noisy_sentence):    
    prompt = f"""
You are required to return the corrected sentence in JSON format. Ensure your response strictly adheres to the JSON structure below.
알아볼 수 있는 단어의 의미를 최대한 살려서 문장을 생성해주세요. 
Examples:

Input:
sentence: 금 시 충격 일단 소국면 주 낙폭 줄고 환도 하"
Output:
{{"corrected_text": "금융시장 충격 일단 소강국면 주가 낙폭 줄고 환율도 하락"}}

Input:
sentence: SKT 유 시스템 5G 드 용 ICT 소 루션 개 발 협 력"
Output:
{{"corrected_text": "SKT유콘시스템 5G 드론용 ICT 솔루션 개발 협력"}}

Input:
sentence: 예스24 우가 랑한 24의 J 가들 산서 시"
Output:
{{"corrected_text": "예스24 우리가 사랑한 24인의 작가들 부산서 전시"}}

generate the corrected text for the following input in the same JSON format:
Input:
sentence: {noisy_sentence}"
Output:
"""
    
    inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=64, pad_token_id=tokenizer.eos_token_id)
    corrected_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract the corrected sentence only, removing additional information
    corrected_sentence = extract_corrected_text(corrected_sentence)
    return corrected_sentence

# Apply the cleaning function to your data
tqdm.pandas()

# Apply the function by passing both 'text' and 'label'
data['clean_text'] = data.progress_apply(lambda row: clean_sentence(row['processed_text']), axis=1)

# Save the cleaned data
data.to_csv(os.path.join(OUTPUT_DIR, 'train_cleaned.csv'), index=False)
