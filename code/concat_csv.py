import pandas as pd

# CSV 파일 불러오기
csv1 = pd.read_csv("/data/ephemeral/home/Yunseo_DCTC/data/expanded_augmented.csv")
csv2 = pd.read_csv("/data/ephemeral/home/Yunseo_DCTC/output/un-noised_augmented/expanded_augmented.csv")

# 필요한 열만 선택 및 열 이름 변경
csv1_selected = csv1[['ID', 'augmented_text', 'target']].rename(columns={'augmented_text': 'text'})
csv2_selected = csv2[['ID', 'augmented_text', 'target']].rename(columns={'augmented_text': 'text'})

# 두 데이터프레임을 행 방향으로 결합
csv3 = pd.concat([csv1_selected, csv2_selected], ignore_index=True)

# 결과 확인 (필요한 경우)
print(csv3.head())

# 새로운 CSV 파일로 저장
csv3.to_csv("last.csv", index=False)
