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
model_name = 'meta-llama/Llama-3.1-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)

# Load your data
data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

label_mapping = {
    0: '생활문화',
    1: '스포츠',
    2: '정치',
    3: '사회',
    4: 'IT과학',
    5: '경제',
    6: '세계',
}

data['label'] = data['target'].map(label_mapping)

def extract_corrected_text(output):
    # 모든 "corrected_text" 항목을 찾아 리스트로 저장
    matches = re.findall(r'"corrected_text":\s*"([^"]+)"', output)
    # 마지막 항목을 반환
    return matches[-1] if matches else None


# Function to clean a single sentence using the model
def clean_sentence(noisy_sentence, label):    
    prompt = f"""
You are required to return the corrected sentence in JSON format. Ensure your response strictly adheres to the JSON structure below.

Examples:

Input:
Original: "topic: 경제, sentence: 금U시R 충격 일단 소R국면… 주Z 낙폭 줄고 환A도 하M"
Output:
{{"topic": "경제", "corrected_text": "금융시장 충격 일단 소강국면…주가 낙폭 줄고 환율도 하락"}}

Input:
Original: "topic: IT과학, sentence: SKT#유@콘시스템 5G 드#론용 I!CT 소*루션 개+발 협9력"
Output:
{{"topic": "IT과학", "corrected_text": "SKT유콘시스템 5G 드론용 ICT 솔루션 개발 협력"}}

Input:
Original: "topic: 사회, sentence: 예스24 우D가 S$랑한 24!인의 J@가들 Z산서 T시"
Output:
{{"topic": "사회", "corrected_text": "예스24 우리가 사랑한 24인의 작가들 부산서 전시"}}

Now, understand the meaning of the sentences and correctly create sentence that fit the topic. generate the corrected text for the following input in the same JSON format:
Input:
Original: "topic: {label}, sentence: {noisy_sentence}"
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
data['clean_text'] = data.progress_apply(lambda row: clean_sentence(row['text'], row['label']), axis=1)

# Save the cleaned data
data.to_csv(os.path.join(OUTPUT_DIR, 'train_cleaned.csv'), index=False)
