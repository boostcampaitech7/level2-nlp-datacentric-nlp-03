import pandas as pd
import os
from deep_translator import GoogleTranslator

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
OUTPUT_DIR = os.path.join(BASE_DIR, '../output')

# CSV 파일 읽기
df = pd.read_csv(os.path.join(DATA_DIR, "typos.csv"))

# 백번역 함수 정의
def back_translate(text, src='ko', mid='en'):
    try:
        # 원본 텍스트를 중간 언어로 번역 (예: 한국어 -> 영어)
        translated = GoogleTranslator(source=src, target=mid).translate(text)
        
        # 중간 언어에서 다시 원본 언어로 번역 (예: 영어 -> 한국어)
        back_translated = GoogleTranslator(source=mid, target=src).translate(translated)
        
        return back_translated
    except Exception as e:
        print(f"번역 중 오류 발생: {text}\n에러 메시지: {e}")
        return text

# 'processed_text' 열에 백번역 적용
df['back_translated_text'] = df['clean_text'].apply(lambda x: back_translate(str(x)))

# 결과를 새로운 CSV 파일로 저장
df.to_csv(os.path.join(OUTPUT_DIR, "back_translated_data.csv"), index=False)

# 결과 확인
print(df[['ID', 'processed_text', 'back_translated_text']])
