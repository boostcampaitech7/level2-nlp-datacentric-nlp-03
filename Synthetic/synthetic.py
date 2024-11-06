import os 
import pandas as pd
from transformers import LlamaTokenizerFast, AutoModelForCausalLM
import torch
from tqdm import tqdm
import re
import time

# LLaMA 모델과 토크나이저 로드
model_name = "NCSOFT/Llama-VARCO-8B-Instruct"  
tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

BASE_DIR = os.getcwd()
output_file = f"{BASE_DIR}/generated_news_titles.csv"

data = pd.read_csv(os.path.join(BASE_DIR, "last.csv"))

while True:
    for target_value in tqdm(range(7), desc="Generating articles for each target"):
        target_data = data[data['target'] == target_value].sample(n=10)
        
        prompts = """
프롬프트:
1. 주어진 뉴스 기사들의 공통된 주제와 주요 정보를 분석하세요.
2. 기사들 간의 유사점에 주목하여 기사들이 집중하고 있는 공통된 핵심을 찾아내세요.
3. 마지막으로, 찾아낸 하나의 공통점을 바탕으로 주어진 기사들의 내용을 복사하거나 재사용하지 말고 완전히 새로운 뉴스 기사 10개를 작성하세요.

예시
새로운 뉴스 기사: 
1. 인공지능 기반의 스마트 홈 기기들이 시장 점유율을 높이는데 기여
2. IT 기업들, 자체 AI 플랫폼 구축에 적극 나서
3. AI 기술 도입으로 업무 효율성 향상 기대감 상승

---
프롬프트의 지시사항을 수행하시오. 

주어진 뉴스 기사:
""" + \
''.join(f"- {text}\n" for text in target_data['text']) + \
"""
새로운 뉴스 기사:
"""

        input_ids = tokenizer(prompts, return_tensors="pt").input_ids.to("cuda")

        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=300, temperature=0.9)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(generated_text)
        
        try:
            if "새로운 뉴스 기사:" in generated_text:
                new_articles_text = generated_text.split("새로운 뉴스 기사:")[-1]
            else:
                continue

            lines = new_articles_text.strip().split('\n')

            # 정규표현식을 사용하여 번호로 시작하는 줄만 추출합니다.
            pattern = re.compile(r'^\s*\d+\.\s*(.*)')
            cleaned_titles = []
            for line in lines:
                match = pattern.match(line)
                if match:
                    title = match.group(1).strip()
                    # 불필요한 문자 제거
                    title = title.strip('[]"\',')
                    if title:
                        cleaned_titles.append(title)
                else:
                    # 번호로 시작하지 않는 줄을 만나면 반복을 중단합니다.
                    break
        except (IndexError, ValueError) as e:
            print(f"Error parsing generated titles: {e}")
            cleaned_titles = ["Generated title error"]

        # 생성된 제목들을 데이터프레임으로 변환합니다.
        output_df = pd.DataFrame({
            "text": cleaned_titles,
            "target": [target_value]*len(cleaned_titles)
        })

        # 데이터프레임을 CSV 파일에 추가합니다.
        output_df.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))

        torch.cuda.empty_cache()
    time.sleep(1)
