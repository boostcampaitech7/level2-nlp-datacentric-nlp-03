import pandas as pd
import re
import os
import logging

class Noise_isolation:
    def __init__(self, input_file, output_file, non_korean_ratio_lower_threshold=0.1, non_korean_ratio_upper_threshold=0.3):
        """
        Noise_isolation
     클래스는 주어진 CSV 파일에서 텍스트 데이터를 처리하여
        한국어가 아닌 문자 비율을 계산하고, 지정된 비율 범위에 해당하는 데이터를 필터링합니다.

        :param input_file: 입력 CSV 파일 경로
        :param output_file: 출력 CSV 파일 경로
        :param non_korean_ratio_lower_threshold: 한국어가 아닌 문자 비율 하한값
        :param non_korean_ratio_upper_threshold: 한국어가 아닌 문자 비율 상한값
        """
        # 로깅 설정
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        self.input_file = input_file
        self.output_file = output_file
        self.non_korean_ratio_lower_threshold = non_korean_ratio_lower_threshold
        self.non_korean_ratio_upper_threshold = non_korean_ratio_upper_threshold
        self.pattern = r'(?<![A-Z])[A-Z](?![A-Z])|[^ㄱ-ㅎ가-힣\u4E00-\u9FFF0-9\sA-Z\u2026·∼]'

        self.df = None  # 원본 데이터프레임
        self.filtered_df = None  # 필터링된 데이터프레임

    def replace_ellipsis(self, text):
        """
        주어진 문자열에서 '...'을 '…'으로 대체합니다.

        :param text: 입력 문자열
        :return: 변환된 문자열
        """
        if isinstance(text, str):
            return text.replace('...', '…')
        return text

    def extract_non_korean_chars(self, text):
        """
        주어진 문자열에서 정규 표현식 패턴에 매칭되는 문자들만 추출하여 반환합니다.

        :param text: 입력 문자열
        :return: 추출된 문자들의 문자열
        """
        if isinstance(text, str):
            matches = re.findall(self.pattern, text)
            return ''.join(matches)
        return ''

    def calculate_non_korean_ratio_vectorized(self):
        """
        벡터화된 방법으로 한국어가 아닌 문자 비율을 계산합니다.
        """
        # 'non_korean_count'는 'non_kor' 열의 길이로 계산
        self.df['non_korean_count'] = self.df['non_kor'].str.len()
        # 전체 문자 수 계산
        self.df['total_chars'] = self.df['text'].str.len()
        # 비율 계산
        self.df['non_korean_ratio'] = (self.df['non_korean_count'] / self.df['total_chars']).round(4)
        # 불필요한 중간 열 제거
        self.df.drop(['non_korean_count', 'total_chars'], axis=1, inplace=True)
        logging.info("한국어가 아닌 문자 비율을 계산하여 'non_korean_ratio' 열에 추가했습니다.")

    def read_data(self):
        """
        CSV 파일을 읽어옵니다.
        """
        try:
            self.df = pd.read_csv(self.input_file, encoding='utf-8')
            logging.info(f"입력 파일 '{self.input_file}'을 성공적으로 읽었습니다.")
            if 'text' not in self.df.columns:
                logging.error("입력 데이터에 'text' 열이 존재하지 않습니다.")
                raise ValueError("입력 데이터에 'text' 열이 존재하지 않습니다.")
        except FileNotFoundError:
            logging.error(f"입력 파일을 찾을 수 없습니다: {self.input_file}")
            raise
        except Exception as e:
            logging.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
            raise

    def clean_data(self):
        """
        데이터 전처리를 수행합니다.
        """
        # 'text' 열의 NaN 값을 빈 문자열로 대체
        self.df['text'] = self.df['text'].fillna('')
        # 'text' 열의 앞뒤 공백 제거
        self.df['text'] = self.df['text'].str.strip()
        # '...'을 '…'으로 대체
        self.df['text'] = self.df['text'].apply(self.replace_ellipsis)

    def process_data(self):
        """
        데이터 처리 및 필터링을 수행합니다.
        """
        # 'non_kor' 열에 패턴에 매칭되는 문자들만 추출
        self.df['non_kor'] = self.df['text'].apply(self.extract_non_korean_chars)
        # 한국어가 아닌 문자 비율 계산
        self.calculate_non_korean_ratio_vectorized()
        # 지정된 비율 범위에 해당하는 데이터 필터링
        self.filtered_df = self.df[
            (self.df['non_korean_ratio'] > self.non_korean_ratio_lower_threshold) &
            (self.df['non_korean_ratio'] <= self.non_korean_ratio_upper_threshold)
        ]
        logging.info(f"한국어가 아닌 문자 비율이 {self.non_korean_ratio_lower_threshold} 이상 {self.non_korean_ratio_upper_threshold} 이하인 문장을 필터링했습니다. 총 {len(self.filtered_df)}개의 문장이 추출되었습니다.")

    def save_data(self):
        """
        필터링된 데이터를 CSV 파일로 저장합니다.
        """
        # 출력 디렉토리 생성
        output_dir = os.path.dirname(self.output_file)
        os.makedirs(output_dir, exist_ok=True)

        try:
            self.filtered_df.to_csv(self.output_file, index=False, encoding='utf-8-sig')
            logging.info(f"필터링된 데이터가 '{self.output_file}' 파일에 저장되었습니다.")
        except Exception as e:
            logging.error(f"파일을 저장하는 중 오류가 발생했습니다: {e}")
            raise

    def run(self, save=False):
        """
        전체 프로세스를 실행합니다.
        """
        self.read_data()
        self.clean_data()
        self.process_data()
        if save:
            self.save_data()

        return self.filtered_df

if __name__ == "__main__":
    # 입력 및 출력 파일 경로 설정
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, '../data')
    OUTPUT_DIR = os.path.join(BASE_DIR, '../output')

    input_file = os.path.join(DATA_DIR, 'train.csv')
    output_file = os.path.join(OUTPUT_DIR, 'noise_isolated.csv')

    # 클래스 인스턴스 생성
    korean_text_filter = Noise_isolation(
        input_file=input_file,
        output_file=output_file,
        non_korean_ratio_lower_threshold=0.1,  # 필요에 따라 값 조정
        non_korean_ratio_upper_threshold=0.36   # 필요에 따라 값 조정
    )

    # 프로세스 실행
    df = korean_text_filter.run(save=True)

    print(df.head())
