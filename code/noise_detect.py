import os
import re
import pandas as pd

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
OUTPUT_DIR = os.path.join(BASE_DIR, '../output')

df = pd.read_csv(os.path.join(DATA_DIR, 'rmkor_filtered.csv'))

def replace_non_korean_with_space(text):
    """
    주어진 문자열에서 한국어와 연속된 대문자 알파벳을 제외한 모든 문자를 공백으로 대체합니다.
    대문자와 숫자가 함께 있는 경우는 공백으로 대체합니다.
    """
    if isinstance(text, str):
        # 유지할 패턴: 한글, 두 글자 이상의 연속된 대문자 (숫자 제외)
        pattern_to_keep = r'[ㄱ-ㅎ가-힣]+|[A-Z]{2,}'
        # 전체 문자열에서 유지할 패턴에 매칭되는 부분을 찾습니다.
        matches = list(re.finditer(pattern_to_keep, text))
        result = []
        last_end = 0
        for match in matches:
            start, end = match.span()
            # 매칭되지 않은 부분을 공백으로 채웁니다.
            if start > last_end:
                # 매칭되지 않은 부분에서 공백이 아닌 문자는 공백으로 대체
                non_matched = text[last_end:start]
                non_matched_spaces = re.sub(r'\S', ' ', non_matched)
                result.append(non_matched_spaces)
            # 매칭된 부분은 그대로 추가합니다.
            result.append(match.group())
            last_end = end
        # 마지막 매칭 이후의 부분 처리
        if last_end < len(text):
            non_matched = text[last_end:]
            non_matched_spaces = re.sub(r'\S', ' ', non_matched)
            result.append(non_matched_spaces)
        # 결과를 합칩니다.
        text_with_spaces = ''.join(result)
        # 연속된 공백을 하나의 공백으로 변환하고, 양쪽 공백 제거
        text_with_spaces = re.sub(r'\s+', ' ', text_with_spaces).strip()
        return text_with_spaces
    return text


# 'text' 열에서 한국어 외 문자 공백 처리
df['processed_text'] = df['text'].apply(replace_non_korean_with_space)

# 결과 출력
print(df[['text', 'processed_text']])


df.to_csv(os.path.join(OUTPUT_DIR, 'clean_df.csv'), index=False)