from noise_isolation import Noise_isolation
from augmentation import Augmentation
from LED import LabelErrorCorrector
import os
import numpy as np

# 입력 및 출력 파일 경로 설정
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
OUTPUT_DIR = os.path.join(BASE_DIR, '../output')

input_file = os.path.join(DATA_DIR, 'train.csv')
noise_isolated_output_file = os.path.join(OUTPUT_DIR, 'noise_isolated.csv')
augmentation_output_file = os.path.join(OUTPUT_DIR, 'augmented')

# 클래스 인스턴스 생성
unnoised_data = Noise_isolation(
    input_file=input_file,
    output_file=noise_isolated_output_file,
    non_korean_ratio_lower_threshold=-1,
    non_korean_ratio_upper_threshold=0.11
)

noised_data = Noise_isolation(
    input_file=input_file,
    output_file=noise_isolated_output_file,
    non_korean_ratio_lower_threshold=0.1,  
    non_korean_ratio_upper_threshold=0.36  
)

# isolation 실행
denoised_df = unnoised_data.run(save=False)
noised_df = noised_data.run(save=False)

# 노이지 데이터 복원 및 증강 
augmentation = Augmentation(
    input_data=noised_df,
    output_dir=augmentation_output_file
)
augmented_data = augmentation.run_all()



# 증강된 데이터를 바탕으로 Label Error Detection
# 학습 데이터 확인
print("학습 데이터 프레임의 처음 몇 줄:")
print(augmented_data.head())

# 테스트 데이터 확인
print("테스트 데이터 프레임의 처음 몇 줄:")
print(augmented_data.head())

# 학습 데이터에서 텍스트와 라벨 추출
train_texts = augmented_data['augmented_text'].tolist()
train_labels = augmented_data['target'].values

# 테스트 데이터에서 텍스트와 라벨, 인덱스 추출
test_texts = unnoised_data['text'].tolist()
test_labels = unnoised_data['target'].values
test_indices = unnoised_data.index.tolist()

# LabelErrorCorrector 클래스 초기화
lec = LabelErrorCorrector(
    model_name='klue/roberta-base',  
    num_labels=7,  # 0~6까지 총 7개 클래스
    batch_size=16,
    epochs=3,
    learning_rate=2e-5,
    random_state=42
)

# 학습용 데이터로 모델 학습
lec.train_model(train_texts, train_labels)

# 테스트용 데이터에서 라벨 에러 검출 및 수정
label_errors = lec.detect_label_errors(test_texts, test_labels)
y_corrected = lec.correct_labels(test_texts, test_labels)

# 수정되지 않은 데이터와 수정된 데이터 결합
corrected_data = unnoised_data.copy()
corrected_data['corrected_target'] = y_corrected
corrected_data['corrected'] = np.where(test_labels == y_corrected, 'No', 'Yes')

relabeled_data = corrected_data[['ID', 'text', 'target']].copy()
relabeled_data['target'] = corrected_data['corrected_target']  # 수정된 라벨로 덮어쓰기
relabeled_data.to_csv("corrected_test_data.csv", index=False)

print("수정된 데이터가 'corrected_test_data.csv'에 저장되었습니다.")

corrected_indices = np.where(y_corrected != test_labels)[0]
print(f"수정된 라벨 개수: {len(corrected_indices)}")
print(f"수정된 라벨 인덱스: {corrected_indices}")



## 라벨링 된 unnoised data 증강
augmentation_labeled_unnoised = Augmentation(
    input_data=relabeled_data,
    output_dir=augmentation_output_file
)
labeled_unnoised_augmented = augmentation_labeled_unnoised.run_all()


# 증강된 labeled unnoised data + 증강 및 복원된 noised data
import pandas as pd

# 필요한 열만 선택 및 열 이름 변경
augmented_data_selected = augmented_data[['ID', 'augmented_text', 'target']].rename(columns={'augmented_text': 'text'})
labeled_unnoised_augmented_selected = labeled_unnoised_augmented[['ID', 'augmented_text', 'target']].rename(columns={'augmented_text': 'text'})

# 두 데이터프레임을 행 방향으로 결합
last = pd.concat([augmented_data_selected, labeled_unnoised_augmented_selected], ignore_index=True)

# 결과 확인 (필요한 경우)
print(last.head())

# 새로운 CSV 파일로 저장
last.to_csv("last.csv", index=False)