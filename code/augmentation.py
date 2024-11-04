import os
import re
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from huggingface_hub import login
from deep_translator import GoogleTranslator
from koeda import EDA  
from sklearn.utils import shuffle


class Augmentation:
    def __init__(self, input_data, output_dir, seed=456):
        """
        Augmentation 클래스는 데이터 증강을 위한 다양한 기법을 제공합니다.

        :param input_file: 입력 CSV 파일 경로 또는 DataFrame
        :param output_dir: 출력 파일을 저장할 디렉토리 경로
        :param seed: 재현성을 위한 랜덤 시드 값
        """
        self.input_data = input_data
        self.output_dir = output_dir
        self.seed = seed

        # 재현성을 위한 시드 설정
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # 디바이스 설정
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # 데이터 로드
        if isinstance(self.input_data, str):
            self.data = pd.read_csv(self.input_data)
        elif isinstance(self.input_data, pd.DataFrame):
            self.data = self.input_data.copy()
        else:
            raise ValueError("input_data는 파일 경로(str) 또는 DataFrame이어야 합니다.")

        # 출력 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)

        # koeda EDA 초기화
        self.eda = EDA(morpheme_analyzer="Okt", alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, prob_rd=0.1)

    def noise_filtering(self):
        """
        노이즈 필터링을 수행하는 함수입니다.
        정규식을 사용하여 텍스트에서 한글, 숫자, 연속된 대문자(2자 이상), 한자, 그리고 띄어쓰기를 제외한 문자를 제거합니다.
        """
        import re

        def clean_text(text):
            if isinstance(text, str):
                # 허용된 패턴: 한글, 숫자, 연속된 대문자(2자 이상), 한자, 띄어쓰기
                pattern = r'[가-힣]+|[0-9]+|[A-Z]{2,}|[\u4E00-\u9FFF]+|\s+'
                matches = re.findall(pattern, text)
                cleaned_text = ''.join(matches)
                return cleaned_text
            return ''

        # 'text' 열을 처리하여 필요한 문자만 남김
        self.data['processed_text'] = self.data['text'].apply(clean_text)

        # 결과를 CSV 파일로 저장
        self.data.to_csv(os.path.join(self.output_dir, 'noise_filtered.csv'), index=False)
        print("노이즈 필터링이 완료되었습니다. 'noise_filtered.csv'에 저장되었습니다.")

    def noise_recovery(self, model_name='Bllossom/llama-3.2-Korean-Bllossom-3B', hf_token=None):
        """
        노이즈 복구를 수행하는 함수입니다.
        LLM을 사용하여 노이즈가 있는 문장을 복구합니다.

        :param model_name: 사용할 모델의 이름
        :param hf_token: Hugging Face 토큰 (필요한 경우)
        """
        if hf_token:
            login(hf_token)

        # 모델과 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        def extract_corrected_text(output):
            # 모든 "corrected_text" 항목을 찾아 리스트로 저장
            matches = re.findall(r'"corrected_text":\s*"([^"]+)"', output)
            # 마지막 항목을 반환
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
            # Extract the corrected sentence only, removing additional information
            corrected_sentence = extract_corrected_text(corrected_sentence)
            return corrected_sentence

        # 노이즈 복구 적용
        tqdm.pandas()
        self.data['recovered_text'] = self.data['processed_text'].progress_apply(clean_sentence)

        # 결과를 CSV 파일로 저장
        self.data.to_csv(os.path.join(self.output_dir, 'noise_recovered.csv'), index=False)
        print("노이즈 복구가 완료되었습니다. 'noise_recovered.csv'에 저장되었습니다.")

    def typos_corrector(self, model_name="j5ng/et5-typos-corrector"):
        """
        맞춤법 교정을 수행하는 함수입니다.
        사전 훈련된 맞춤법 교정 모델을 사용합니다.

        :param model_name: 사용할 모델의 이름
        """
        # 모델과 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        def correct_spelling(text):
            inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
            corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return corrected_text

        # 맞춤법 교정 적용
        tqdm.pandas()
        self.data['typos_corrected_text'] = self.data['processed_text'].progress_apply(
            lambda x: correct_spelling(str(x))
        )

        # 결과를 CSV 파일로 저장
        self.data.to_csv(os.path.join(self.output_dir, 'typos_corrected.csv'), index=False)
        print("맞춤법 교정이 완료되었습니다. 'typos_corrected.csv'에 저장되었습니다.")

    def back_translation(self, src='ko', mid='en'):
        """
        백번역을 수행하는 함수입니다.
        텍스트를 중간 언어로 번역한 후 다시 원래 언어로 번역합니다.

        :param src: 원본 언어 (기본값: 한국어)
        :param mid: 중간 언어 (기본값: 영어)
        """
        def back_translate(text):
            try:
                # 원본 텍스트를 중간 언어로 번역
                translated = GoogleTranslator(source=src, target=mid).translate(text)
                # 중간 언어에서 다시 원본 언어로 번역
                back_translated = GoogleTranslator(source=mid, target=src).translate(translated)
                return back_translated
            except Exception as e:
                print(f"번역 중 오류 발생: {text}\n에러 메시지: {e}")
                return text

        # 백번역 적용
        tqdm.pandas()
        self.data['back_translated_text'] = self.data['processed_text'].progress_apply(
            lambda x: back_translate(str(x))
        )

        # 결과를 CSV 파일로 저장
        self.data.to_csv(os.path.join(self.output_dir, 'back_translated.csv'), index=False)
        print("백번역이 완료되었습니다. 'back_translated.csv'에 저장되었습니다.")

    def easy_data_augmentation(self, num_aug=1):
        """
        EDA 기법을 koeda 라이브러리를 사용하여 적용하는 함수입니다.

        :param num_aug: 각 기법별로 생성할 문장 수
        """
        tqdm.pandas()
        # 새로운 열을 초기화
        self.data['eda_sr'] = ''
        self.data['eda_ri'] = ''
        self.data['eda_rs'] = ''
        self.data['eda_rd'] = ''

        for idx, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            text = row['processed_text']

            # Synonym Replacement (SR)
            sr_augmented = self.eda(text, p=(0.1, 0, 0, 0))
            self.data.at[idx, 'eda_sr'] = sr_augmented

            # Random Insertion (RI)
            ri_augmented = self.eda(text, p=(0, 0.1, 0, 0))
            self.data.at[idx, 'eda_ri'] = ri_augmented

            # Random Swap (RS)
            rs_augmented = self.eda(text, p=(0, 0, 0.1, 0))
            self.data.at[idx, 'eda_rs'] = rs_augmented

            # Random Deletion (RD)
            rd_augmented = self.eda(text, p=(0, 0, 0, 0.1))
            self.data.at[idx, 'eda_rd'] = rd_augmented

        # 결과를 CSV 파일로 저장
        self.data.to_csv(os.path.join(self.output_dir, 'eda_augmented.csv'), index=False)
        print("EDA 기법이 완료되었습니다. 'eda_augmented.csv'에 저장되었습니다.")

    def melt_columns(self):
        """
        모든 증강된 열을 하나의 열로 변환하여 저장하는 함수입니다.
        """
        # 증강된 열 목록
        columns_to_expand = [
            'processed_text', 'recovered_text', 'typos_corrected_text', 
            'back_translated_text', 'eda_sr', 'eda_ri', 'eda_rs', 'eda_rd'
        ]

        # 각 열을 별도의 행으로 확장하여 하나의 열로 변환
        expanded_data = self.data.melt(
            id_vars=[col for col in self.data.columns if col not in columns_to_expand],
            value_vars=columns_to_expand,
            var_name='augmentation_type',
            value_name='augmented_text'
        ).dropna(subset=['augmented_text'])  # 값이 있는 데이터만 남기기

        # 데이터 셔플링 (재현성을 위해 random_state 설정)
        expanded_data = shuffle(expanded_data, random_state=self.seed).reset_index(drop=True)

        # 결과를 CSV 파일로 저장
        expanded_data.to_csv(os.path.join(self.output_dir, 'expanded_augmented.csv'), index=False)
        print("모든 증강된 열을 하나의 열로 변환하여 'expanded_augmented.csv'에 저장되었습니다.")


    def run_all(self, hf_token=None):
        """
        모든 증강 기법을 순차적으로 실행하는 함수입니다.

        :param hf_token: Hugging Face 토큰 (노이즈 복구에 필요)
        """
        self.noise_filtering()
        self.noise_recovery(hf_token=hf_token)
        self.typos_corrector()
        self.back_translation()
        self.easy_data_augmentation()
        print("모든 증강 기법이 완료되었습니다.")

        self.melt_columns()

if __name__ == "__main__":
    # 입력 및 출력 파일 경로 설정
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, '../data')
    OUTPUT_DIR = os.path.join(BASE_DIR, '../output')

    input_file = os.path.join(DATA_DIR, 'corrected_test_data.csv')
    output_dir = os.path.join(OUTPUT_DIR, 'un-noised_augmented')

    # 클래스 인스턴스 생성
    augmentation = Augmentation(
        input_data=input_file,
        output_dir=output_dir,
    )

    # 프로세스 실행
    augmentation.run_all()
