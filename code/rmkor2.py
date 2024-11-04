import pandas as pd
import re
import os
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def replace_ellipsis(text):
    """
    주어진 문자열에서 '...'을 '…'으로 대체합니다.
    """
    if isinstance(text, str):
        return text.replace('...', '…')
    return text

def extract_non_korean_chars(text, pattern):
    """
    주어진 문자열에서 정규 표현식 패턴에 매칭되는 문자들만 추출하여 반환합니다.
    """
    if isinstance(text, str):
        matches = re.findall(pattern, text)
        return ''.join(matches)
    return ''

def calculate_non_korean_ratio_vectorized(df):
    """
    벡터화된 방법으로 한국어가 아닌 문자 비율을 계산합니다.
    공백, 한자, 숫자, 두 개 이상 연속된 영어 문자, '…'과 '...'을 제외하고 계산합니다.
    """
    # 'non_korean_count'는 'rmkor' 열의 길이로 계산
    df['non_korean_count'] = df['rmkor'].str.len()
    # 전체 문자 수 계산
    df['total_chars'] = df['text'].str.len()
    # 비율 계산
    df['non_korean_ratio'] = (df['non_korean_count'] / df['total_chars']).round(4)
    # 불필요한 중간 열 제거
    df.drop(['non_korean_count', 'total_chars'], axis=1, inplace=True)
    logging.info("한국어가 아닌 문자 비율을 계산하여 'non_korean_ratio' 열에 추가했습니다.")

# CSV 파일 경로 설정
output_file = 'rmkor_filtered.csv'  # 출력 파일 이름 (필터링된 데이터)
NON_KOREAN_RATIO_LOWER_THRESHOLD = 0.1   # 필터링 기준 하한
NON_KOREAN_RATIO_UPPER_THRESHOLD = 0.3  # 필터링 기준 상한

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
OUTPUT_DIR = os.path.join(BASE_DIR, '../output')

input_file = os.path.join(DATA_DIR, 'train.csv')
output_path = os.path.join(OUTPUT_DIR, output_file)

# 출력 디렉토리 생성 (존재하지 않으면)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 데이터 유효성 검사: 'text' 열이 존재하는지 확인
try:
    df = pd.read_csv(input_file, encoding='utf-8')
    logging.info(f"입력 파일 '{input_file}'을 성공적으로 읽었습니다.")
    if 'text' not in df.columns:
        logging.error("입력 데이터에 'text' 열이 존재하지 않습니다.")
        exit(1)
except FileNotFoundError:
    logging.error(f"입력 파일을 찾을 수 없습니다: {input_file}")
    exit(1)
except Exception as e:
    logging.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
    exit(1)

# 'text' 열의 NaN 값을 빈 문자열로 대체 (선택 사항)
df['text'] = df['text'].fillna('')

# 'text' 열의 앞뒤 공백 제거 (선택 사항)
df['text'] = df['text'].str.strip()

# 'text' 열에서 '...'을 '…'으로 대체
df['text'] = df['text'].apply(replace_ellipsis)

# 정규 표현식 패턴 정의
pattern = r'(?<![A-Z])[A-Z](?![A-Z])|[^ㄱ-ㅎ가-힣\u4E00-\u9FFF0-9\sA-Z\u2026·∼]'

# 'rmkor' 열에 패턴에 매칭되는 문자들만 추출
df['rmkor'] = df['text'].apply(lambda x: extract_non_korean_chars(x, pattern))

# 벡터화된 방법으로 한국어가 아닌 문자 비율 계산
calculate_non_korean_ratio_vectorized(df)

# 한국어가 아닌 문자 비율이 기준 이상 0.3 이상 0.35 이하인 문장 필터링
filtered_df = df[
    (df['non_korean_ratio'] > NON_KOREAN_RATIO_LOWER_THRESHOLD) & 
    (df['non_korean_ratio'] <= NON_KOREAN_RATIO_UPPER_THRESHOLD)
]
logging.info(f"한국어가 아닌 문자 비율이 {NON_KOREAN_RATIO_LOWER_THRESHOLD} 이상 {NON_KOREAN_RATIO_UPPER_THRESHOLD} 이하인 문장을 필터링했습니다. 총 {len(filtered_df)}개의 문장이 추출되었습니다.")

try:
    filtered_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logging.info(f"필터링된 데이터가 '{output_path}' 파일에 저장되었습니다.")
except Exception as e:
    logging.error(f"파일을 저장하는 중 오류가 발생했습니다: {e}")
    exit(1)
