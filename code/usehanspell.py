from pykospacing import Spacing
import pandas as pd
import os 
import matplotlib.pyplot as plt

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
OUTPUT_DIR = os.path.join(BASE_DIR, '../output')

from pykospacing import Spacing
import pandas as pd
import numpy as np
from tqdm import tqdm

# 예시 데이터프레임 생성
# data = {
#     'ID': ['ynat-v1_train_00009', 'ynat-v1_train_00010'],
#     'text': ['듀얼심 아이폰 하반기 출시설 솔솔…알뜰폰 기대감', 'oi 매력 R모h츠a열#w3약 >l·주가 고Q/진']
# }

df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

# 예시 데이터프레임 생성
# data = {
#     'ID': ['ynat-v1_train_00009', 'ynat-v1_train_00010'],
#     'text': ['듀얼심 아이폰 하반기 출시설 솔솔…알뜰폰 기대감', 'oi 매력 R모h츠a열#w3약 >l·주가 고Q/진']
# }
# df = pd.DataFrame(data)

# PyKoSpacing 초기화
spacing = Spacing()

# 정규화된 오류 비율 계산 함수
def calculate_normalized_error_ratio(text):
    try:
        # 원본 텍스트와 띄어쓰기 교정된 텍스트 비교
        corrected_text = spacing(text)
        
        # 원본과 교정된 텍스트의 차이를 비교하여 오류 개수 계산
        num_changes = sum(1 for a, b in zip(text, corrected_text) if a != b)
        
        # 오류 개수를 문장 길이의 제곱근으로 정규화
        normalized_error_ratio = num_changes / np.sqrt(len(text))
        return normalized_error_ratio
    except Exception as e:
        print(f"Error processing text: {text}, Error: {e}")
        return None  # 오류 발생 시 None으로 처리

# 정규화된 오류 비율을 계산하여 데이터프레임에 추가

tqdm.pandas()
df["normalized_error_ratio"] = df["text"].progress_apply(calculate_normalized_error_ratio)

# print(df[['ID', 'text', 'normalized_error_ratio']])

# 결과에서 None 값을 제외하고 시각화를 위한 데이터 준비
df_filtered = df.dropna(subset=["normalized_error_ratio"])

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.scatter(df["ID"], df["normalized_error_ratio"], marker="o", color="b")
plt.axhline(y=0.1, color="r", linestyle="--", label="Threshold (0.1)")
plt.xlabel("Text ID")
plt.ylabel("Normalized Error Ratio")
plt.title("Normalized Error Ratio for Each Text")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

# 그래프 저장
output_path = "/data/ephemeral/home/Yunseo_DCTC/output/normalized_error_ratio_plot.png"
plt.savefig(output_path)

df.to_csv(os.path.join(OUTPUT_DIR, 'spell.csv'), index=False)