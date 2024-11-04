import os
import pandas as pd
import numpy as np
from cleanlab.classification import CleanLearning
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

# 데이터 로드
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
OUTPUT_DIR = os.path.join(BASE_DIR, '../output')

data = pd.read_csv(os.path.join(DATA_DIR, 'back_translated_data.csv'))

# 펼칠 열 목록
columns_to_expand = ['processed_text', 'clean_text', 'corrected_text', 'back_translated_text']

# 각 열을 별도의 행으로 확장
expanded_data = data.melt(
    id_vars=[col for col in data.columns if col not in columns_to_expand],
    value_vars=columns_to_expand,
    var_name='text_type',
    value_name='expanded_text'
).dropna(subset=['expanded_text'])

# 데이터 섞기 (옵션)
expanded_data = shuffle(expanded_data, random_state=42).reset_index(drop=True)

# 라벨과 텍스트 추출
texts = expanded_data['expanded_text']
labels = expanded_data['target']

# 결측치 제거 및 인덱스 재설정
mask = labels.notnull() & texts.notnull()
texts = texts[mask].reset_index(drop=True)
labels = labels[mask].reset_index(drop=True)

# 텍스트 데이터 전처리 (TF-IDF 벡터화)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# 라벨 인코딩 수행
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

# y_encoded가 numpy array인지 확인
if not isinstance(y_encoded, np.ndarray):
    y_encoded = np.array(y_encoded)

# 모델 초기화
base_model = RandomForestClassifier(random_state=42)

# CleanLearning 사용
clf = CleanLearning(clf=base_model)
clf.fit(X, y_encoded)

# 라벨 오류가 의심되는 인덱스 식별 (X와 labels를 명시적으로 전달)
label_issues = clf.find_label_issues(X=X, labels=y_encoded)

# 라벨 오류가 있는 데이터 확인
print("라벨 오류가 의심되는 데이터:")
expanded_data_with_issues = expanded_data.iloc[label_issues.index].copy()
expanded_data_with_issues['corrected_target'] = label_encoder.inverse_transform(
    clf.predict(X)[label_issues.index]
)

print(expanded_data_with_issues[['ID', 'expanded_text', 'target', 'corrected_target']])

# 결과 저장
expanded_data_with_issues.to_csv(os.path.join(OUTPUT_DIR, 'data_with_corrected_labels.csv'), index=False)
print("라벨 수정이 완료된 데이터가 'data_with_corrected_labels.csv'에 저장되었습니다.")
