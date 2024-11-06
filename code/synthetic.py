from transformers import LlamaTokenizerFast, AutoModelForCausalLM, pipeline
import pandas as pd
import torch

# LLaMA 모델과 토크나이저 로드
model_name = "NCSOFT/Llama-VARCO-8B-Instruct"  # 실제 사용 모델 이름으로 변경
tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

# text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

file_path = "/data/ephemeral/home/level2-nlp-datacentric-nlp-03/code/last.csv"  # 사용자가 제공한 데이터 파일 위치
data = pd.read_csv(file_path)

# target이 0인 문장만 추출
target_zero_data = data[data['target'] == 0].sample(n=10, random_state=3)

prompts = f"""
프롬프트:
1. 주어진 뉴스 기사들의 공통된 주제와 주요 정보를 분석하세요.
2. 기사들 간의 유사점에 주목하여 기사들이 집중하고 있는 공통된 핵심을 찾아내세요.
3. 마지막으로, 찾아낸 공통점을 바탕으로 새로운 뉴스 기사 5개를 작성하세요. 이 기사는 다른 기사들과 어울리는 주제와 분위기를 가져야 하며, 주제에 대한 새로운 시각을 제시하거나 기존 정보를 확장하는 형태여야 합니다.

**예시 뉴스 기사**:
- 기후 변화가 농업에 미치는 영향: 작물 생산 감소 우려
- 기후 변화로 인한 폭염 현상, 전 세계 주요 도시들에 경고음 울리다
- 해수면 상승, 섬나라들의 생존 위협 – 대책 마련 시급
- 북극 해빙 감소, 야생 동물 생태계에 미치는 영향

**예시 출력**:
- 공통점: 주요 주제는 기후 변화로 인한 다양한 환경적, 사회적 영향입니다
- 새로운 기사: ["온난화로 인한 모기와 해충 확산", "너무 빨리 뜨거워지는 지구…기후목표 1.5도 내년에 초과 가능성", "온실가스, 또다시 최고치 경신…악순환 직면", ...]

---
프롬프트의 지시사항을 수행하시오.

**주어진 뉴스 기사**
""" + \
''.join(f"- {text}\n" for text in target_zero_data['text']) + \
"""
- 새로운 기사: 
"""

print(prompts)

# 텍스트 인코딩
input_ids = tokenizer(prompts, return_tensors="pt").input_ids.to("cuda")

# 모델 생성 실행
with torch.no_grad():
    outputs = model.generate(input_ids, max_new_tokens=200, num_return_sequences=1, temperature=0.7)

# 결과 디코딩
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)

# # 결과 출력
# for idx, text in enumerate(generated_texts):
#     generated_text = text["generated_text"]
#     # '새로운 기사:' 이후의 내용을 추출합니다.
#     new_articles = generated_text.split("새로운 기사:")[1].strip() if "새로운 기사:" in generated_text else "생성된 기사를 찾을 수 없음"
    
#     print(f"Generated News Articles:\n{new_articles}")

