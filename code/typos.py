from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import re
import pandas as pd
from tqdm import tqdm

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
OUTPUT_DIR = os.path.join(BASE_DIR, '../output')


import torch

def load_model(model_name="j5ng/et5-typos-corrector"):
    """
    모델과 토크나이저를 로드하는 함수입니다.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def correct_spelling(text, tokenizer, model, device):
    """
    주어진 텍스트의 맞춤법을 교정하는 함수입니다.
    """
    inputs = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

def main(input_csv, output_csv):
    # 모델과 토크나이저 로드
    tokenizer, model = load_model()
    
    # GPU 사용 가능 시 GPU 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # CSV 데이터 읽기
    df = pd.read_csv(os.path.join(DATA_DIR, input_csv))
    
    tqdm.pandas()
    # 맞춤법 수정된 텍스트를 저장할 새로운 열 추가
    df['corrected_text'] = df['clean_text'].progress_apply(lambda x: correct_spelling(str(x), tokenizer, model, device))
    
    # 수정된 데이터를 새로운 CSV 파일로 저장
    df.to_csv(os.path.join(OUTPUT_DIR, output_csv), index=False)
    print(f"맞춤법 검사가 완료된 데이터가 '{output_csv}' 파일에 저장되었습니다.")

if __name__ == "__main__":
    input_csv = "train_cleaned.csv"       # 입력 CSV 파일 경로
    output_csv = "typos.csv"  # 출력 CSV 파일 경로
    main(input_csv, output_csv)
