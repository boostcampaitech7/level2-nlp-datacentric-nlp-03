import os
import re
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from collections import defaultdict
from huggingface_hub import login
from deep_translator import GoogleTranslator
from koeda import EDA  
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

class Augmentation:
    def __init__(self, output_dir, seed=2024):
        self.output_dir = output_dir
        self.seed = seed

        # 재현성을 위한 시드 설정
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        # 디바이스 설정
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)

        # koeda EDA 초기화
        self.eda = EDA(morpheme_analyzer="Okt", alpha_sr=0.3, alpha_ri=0.3, alpha_rs=0.3, prob_rd=0.3)

        # 내부적으로 증강된 데이터를 저장할 데이터프레임
        self.augmented_df = None
        self.base_data = None  # 현재 기초 데이터

    def _reset_augmented_df(self):
        # augmented_df를 초기화합니다.
        self.augmented_df = pd.DataFrame(columns=self.base_data.columns)

    def _check_base_data(self, data):
        # 기초 데이터가 변경되었는지 확인하고, 변경되었으면 augmented_df를 초기화합니다.
        if self.base_data is None or not self.base_data.equals(data):
            print("base data가 변경되었습니다.")
            self.base_data = data.copy()
            self._reset_augmented_df()

    def noise_filtering(self, data):
        """
        'text' 열에 노이즈 필터링을 수행합니다.
        """
        self._check_base_data(data)

        def clean_text(text):
            if isinstance(text, str):
                # 허용된 패턴: 한글, 숫자, 연속된 대문자(2자 이상), 한자, 띄어쓰기
                pattern = r'[가-힣]+|[0-9]+|[A-Z]{2,}|[\u4E00-\u9FFF]+|\s+'
                matches = re.findall(pattern, text)
                cleaned_text = ''.join(matches).strip()
                return cleaned_text
            return ''

        augmented_data = data.copy()
        augmented_data['text'] = augmented_data['text'].apply(clean_text)
        self.augmented_df = pd.concat([self.augmented_df, augmented_data], ignore_index=True)

        print("노이즈 필터링이 완료되었습니다.")
        return self.augmented_df

    def naive_augment(self, data):
        """
        original + remove_special_characters + to_lowercase + add_spaces
        """
        self._check_base_data(data)

        def remove_special_characters(text):
            return re.sub(r'[^A-Za-z0-9가-힣 ]+', '', text)

        def to_lowercase(text):
            return text.lower()

        def add_spaces(text):
            return '  '.join(text.split())

        augmented_data = pd.DataFrame(data)
        for func in [remove_special_characters, to_lowercase, add_spaces]:
            temp_data = data.copy()
            temp_data['text'] = temp_data['text'].apply(func)
            augmented_data = pd.concat([augmented_data, temp_data], ignore_index=True)

        self.augmented_df = pd.concat([self.augmented_df, augmented_data], ignore_index=True)
        print("Naive 증강이 완료되었습니다.")
        return self.augmented_df

    def llm_recovery(self, data, model_name='Bllossom/llama-3.2-Korean-Bllossom-3B', hf_token=None):
        """
        'text' 열에 LLM 복구를 수행합니다.
        """
        self._check_base_data(data)

        if hf_token:
            login(hf_token)

        # 모델과 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        def extract_corrected_text(output):
            matches = re.findall(r'"corrected_text":\s*"([^"]+)"', output)
            return matches[-1] if matches else None

        def clean_sentence(noisy_sentence):
            prompt = f"""
You are required to return the corrected sentence in JSON format. Ensure your response strictly adheres to the JSON structure below.
알아볼 수 있는 단어의 의미를 최대한 살려서 문장을 생성해주세요. 
Examples:

Input:
sentence: 금 시 충격 일단 소국면 주 낙폭 줄고 환도 하"
Output:
{{"corrected_text": "금융시장 충격 일단 소강국면 주가 낙폭 줄고 환율도 하락"}}

Input:
sentence: SKT 유 시스템 5G 드 용 ICT 소 루션 개 발 협 력"
Output:
{{"corrected_text": "SKT유콘시스템 5G 드론용 ICT 솔루션 개발 협력"}}

Input:
sentence: 예스24 우가 랑한 24의 J 가들 산서 시"
Output:
{{"corrected_text": "예스24 우리가 사랑한 24인의 작가들 부산서 전시"}}

generate the corrected text for the following input in the same JSON format:
Input:
sentence: {noisy_sentence}"
Output:
"""

            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(self.device)
            outputs = model.generate(**inputs, max_new_tokens=64, pad_token_id=tokenizer.eos_token_id)
            corrected_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
            corrected_sentence = extract_corrected_text(corrected_sentence)
            return corrected_sentence

        augmented_data = data.copy()
        tqdm.pandas()
        augmented_data['text'] = augmented_data['text'].progress_apply(clean_sentence)
        self.augmented_df = pd.concat([self.augmented_df, augmented_data], ignore_index=True)

        print("LLM 복구가 완료되었습니다.")
        return self.augmented_df

    def typos_corrector(self, data, model_name="j5ng/et5-typos-corrector"):
        """
        'text' 열에 맞춤법 교정을 수행합니다.
        """
        self._check_base_data(data)

        # 모델과 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        def correct_spelling(text):
            inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
            corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return corrected_text

        augmented_data = data.copy()
        tqdm.pandas()
        augmented_data['text'] = augmented_data['text'].progress_apply(correct_spelling)
        self.augmented_df = pd.concat([self.augmented_df, augmented_data], ignore_index=True)

        print("맞춤법 교정이 완료되었습니다.")
        return self.augmented_df

    def back_translation(self, data, src='ko', languages=None):
        """
        'text' 열에 백번역을 수행합니다.
        """
        self._check_base_data(data)

        if languages is None:
            languages = [('en', 'English'), ('zh-CN', 'Chinese'), ('de', 'German')]

        augmented_data = pd.DataFrame()
        for mid_lang, lang_name in languages:
            temp_data = data.copy()
            tqdm.pandas(desc=f"Back-translation ({lang_name})")
            def back_translate(text):
                try:
                    translated = GoogleTranslator(source=src, target=mid_lang).translate(text)
                    back_translated = GoogleTranslator(source=mid_lang, target=src).translate(translated)
                    return back_translated
                except Exception as e:
                    print(f"번역 오류 ({lang_name}): {e}")
                    return text
            temp_data['text'] = temp_data['text'].progress_apply(back_translate)
            augmented_data = pd.concat([augmented_data, temp_data], ignore_index=True)

        self.augmented_df = pd.concat([self.augmented_df, augmented_data], ignore_index=True)

        print("Back Translation이 완료되었습니다.")
        return self.augmented_df


    def easy_data_augmentation(self, data, num_aug=1):
        """
        'text' 열에 대해 EDA 증강을 수행합니다.
        """
        # 기존의 기초 데이터가 변경되었는지 확인 및 설정
        self._check_base_data(data)

        # 증강된 데이터를 저장할 딕셔너리 초기화
        expanded_data = defaultdict(list)

        for idx, row in tqdm(data.iterrows(), total=len(data)):
            text = row['text']
            target = row['target']
            record_id = row['ID']

            # 원본 데이터 추가
            expanded_data['ID'].append(record_id)
            expanded_data['text'].append(text)
            expanded_data['target'].append(target)

            # Synonym Replacement (SR)
            sr_augmented = self.eda(text, p=(0.1, 0, 0, 0))
            expanded_data['ID'].append(record_id)
            expanded_data['text'].append(sr_augmented)
            expanded_data['target'].append(target)

            # Random Insertion (RI)
            ri_augmented = self.eda(text, p=(0, 0.1, 0, 0))
            expanded_data['ID'].append(record_id)
            expanded_data['text'].append(ri_augmented)
            expanded_data['target'].append(target)

            # Random Swap (RS)
            rs_augmented = self.eda(text, p=(0, 0, 0.1, 0))
            expanded_data['ID'].append(record_id)
            expanded_data['text'].append(rs_augmented)
            expanded_data['target'].append(target)

            # Random Deletion (RD)
            rd_augmented = self.eda(text, p=(0, 0, 0, 0.1))
            expanded_data['ID'].append(record_id)
            expanded_data['text'].append(rd_augmented)
            expanded_data['target'].append(target)

        # 딕셔너리를 데이터프레임으로 변환하여 augmented_df에 추가
        augmented_df = pd.DataFrame(expanded_data)
        self.augmented_df = pd.concat([self.augmented_df, augmented_df], ignore_index=True)

        print("EDA 증강이 완료되었습니다.")
        return self.augmented_df


if __name__ == "__main__":
    # 입력 및 출력 파일 경로 설정
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, '../data')
    OUTPUT_DIR = os.path.join(BASE_DIR, '../output')

    input_file = os.path.join(DATA_DIR, 'corrected_test_data.csv')
    output_dir = os.path.join(OUTPUT_DIR, 'un-noised_augmented')

    # 클래스 인스턴스 생성
    augmentation = Augmentation(
        output_dir=output_dir,
    )

    # 프로세스 실행
    augmentation.run_all()

