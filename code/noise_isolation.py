import pandas as pd
import re
import os
import logging

class Noise_isolation:
    def __init__(self, input_file, output_dir):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        self.input_file = input_file
        self.output_dir = output_dir

        self.pattern = r'(?<![A-Z])[A-Z](?![A-Z])|[^ㄱ-ㅎ가-힣\u4E00-\u9FFF0-9\sA-Z\u2026·∼]'

        self.df = None  # 원본 데이터프레임
        self.filtered_df = None  # 필터링된 데이터프레임

    # 주어진 문자열에서 '...'을 '…'으로 대체합니다.
    def replace_ellipsis(self, text):
        if isinstance(text, str):
            return text.replace('...', '…')
        return text
    
    # 주어진 문자열에서 정규 표현식 패턴에 매칭되는 문자들만 추출하여 반환합니다.
    def extract_non_korean_chars(self, text):
        if isinstance(text, str):
            matches = re.findall(self.pattern, text)
            return ''.join(matches)
        return ''
    # 한국어가 아닌 문자의 비율을 계산합니다.
    def calculate_non_korean_ratio(self):
        self.df['non_korean_count'] = self.df['non_kor'].str.len() # 한국어가 아닌 문자 'non_kor' 열의 길이로 계산
        
        self.df['total_chars'] = self.df['text'].str.len() # 전체 문자 수 계산
        
        self.df['non_korean_ratio'] = (self.df['non_korean_count'] / self.df['total_chars']).round(4) # 비율 계산
        
        self.df.drop(['non_korean_count', 'total_chars'], axis=1, inplace=True) # 불필요한 중간 열 제거
        logging.info("한국어가 아닌 문자 비율을 계산하여 'non_korean_ratio' 열에 추가했습니다.")

    def calculate_noise_ratio_for_hard(self, text):
        allowed_terms = [
            'SERICEO', 'APTLD', 'K 시리즈', 'KAIST', 'MWC19', 'NBIoT', 'U파손도움', 'VR·AR', 'u3000', '갤럭시A9', '갤럭시S8', '아이폰XR', '태블릿PC', 
            'ALCS', 'ETRI', 'FARC', 'GSMA', 'H7N9', 'KISA', 'OECD', 'RCEP', 'S400', 'S500', 'K시리즈', 'T멤버십', 'UCLA', 'UEFA', 'USTR', 'WNBA', 'ai.x', 
            'e스포츠', '스마트X', '아이폰X', '67P', '75t', 'ACM', 'AFC', 'APT', 'ATM', 'AWS', 'AfD', 'A매치', 'B2B', 'B52', 'BMW', 'BeY', 'CBS', 'CCO', 'CEO', 'CPU', 'CSD', 
            'DMZ', 'EBS', 'ELS', 'EPL', 'ETF', 'F35', 'FFP', 'G20', 'G20', 'HDC', 'IBK', 'IBM', 'ICT', 'IPO', 'IRP', 'ISA', 'IoT', 'KAT', 'KBO', 'KBO', 'KBS', 'KEB', 'KGC', 'KTB', 'K리그', 
            'LGU', 'LNG', 'LPG', 'LTE', 'MBC', 'MLB', 'MMT', 'MOU', 'MVP', 'MWC', 'NBA', 'NEW', 'NHK', 'NHN', 'NSA', 'N여행', 'PVC', 'RFA', 'S10', 'SBS', 'SDI', 'SKB', 'SKT', 'SKT', 'SNS', 
            'TBN', 'TPP', 'U19', 'U20', 'UAE', 'UAR', 'UCC', 'URL', 'USC', 'V30', 'V50', 'V리그', 'WON', 'WTI', 'WTO', 'aix', 'com', 'iOS', 'tbs', '채널A', '%P', '%p', '3D', '3Q', '5G', 
            'A8', 'AG', 'AI', 'AS', 'Be', 'CJ', 'C조', 'D8', 'DB', 'EM', 'ES', 'EU', 'FA', 'FC', 'FK', 'G6', 'G7', 'G8', 'GB', 'GS', 'HF', 'IC', 'IP', 'IS', 'IT', 'KB', 'KT', 'K리', 'K콘', 
            'LA', 'LD', 'LG', 'LH', 'LS', 'ML', 'MS', 'NC', 'NH', 'NO', 'OK', 'PC', 'PF', 'PO', 'PS', 'RD', 'S9', 'SA', 'SK', 'ST', 'SW', 'S펜', 'TI', 'TK', 'TV', 'T맵', 'VR', 'VS', 'WS', 'X6', 'XR', 
            'kt', 'vs', ',', '.', 'm', '|', '·', '…', '⅔', '↑', '→', '↓', '↔', '∼', '③', '④', 'ㆍ', '㎜', '㎝', '㎡', '＋', 'ｍ', '2％'
        ]
        text = text.replace(" ", "")
        for term in allowed_terms:
            text = text.replace(term, "")
        text = re.sub(r'(?<=\d)%', '', text)
        non_korean_text = re.sub(r'[가-힣0-9\u4e00-\u9fff]', '', text)
        return len(non_korean_text) / len(text) if len(text) > 0 else 0

    def read_data(self):
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

    def save_data(self):
        output_dir = os.path.dirname(self.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        try:
            self.filtered_df.to_csv(self.output_dir, index=False, encoding='utf-8-sig')
            logging.info(f"필터링된 데이터가 '{self.output_dir}' 파일에 저장되었습니다.")
        except Exception as e:
            logging.error(f"파일을 저장하는 중 오류가 발생했습니다: {e}")
            raise

    def clean_data(self):
        self.df['text'] = self.df['text'].fillna('')
        self.df['text'] = self.df['text'].str.strip()
        self.df['text'] = self.df['text'].apply(self.replace_ellipsis) # '...'을 '…'으로 대체
        self.df['non_kor'] = self.df['text'].apply(self.extract_non_korean_chars) # 'non_kor' 열에 패턴에 매칭되는 문자들만 추출

    def isolate(self, save=False, type='hard'):
        self.read_data()

        if type == 'soft':
            self.clean_data() # 데이터 클리닝 
            self.calculate_non_korean_ratio() 
            # 지정된 비율 범위에 해당하는 데이터 필터링
            self.noised_df = self.df[
                (self.df['non_korean_ratio'] > 0.1) &
                (self.df['non_korean_ratio'] <= 0.36)
            ]
            self.filtered_df = self.df[
                (self.df['non_korean_ratio'] <= 0.11)
            ]
            logging.info(f"노이즈가 포함된 {len(self.noised_df)}개, 노이즈가 제거된 {len(self.filtered_df)}개의 문장이 추출되었습니다.")
        elif type == 'hard':
            self.df['noise_ratio'] = self.df['text'].apply(self.calculate_noise_ratio_for_hard)

            bins = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1]
            labels = ['0', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8+']
            self.df['noise_group'] = pd.cut(self.df['noise_ratio'], bins=bins, labels=labels, include_lowest=True)

            self.noised_df = self.df[self.df['noise_ratio'] > 0].copy()[['ID', 'text', 'target']]
            self.filtered_df = self.df[self.df['noise_ratio'] == 0].copy()[['ID', 'text', 'target']]

            # self.noised_df[['ID', 'text', 'target']].to_csv("noise_train.csv", index=False)
            # self.filtered_df[['ID', 'text', 'target']].to_csv("clean_train.csv", index=False)
        else:
            print("그런건 없습니다.")

        if save:
            self.save_data()

        return self.noised_df, self.filtered_df

if __name__ == "__main__":
    # 입력 및 출력 파일 경로 설정
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, '../data')
    OUTPUT_DIR = os.path.join(BASE_DIR, '../output')

    input_file = os.path.join(DATA_DIR, 'train.csv')
    output_dir = os.path.join(OUTPUT_DIR, 'noise_isolated_test.csv')

    noise_isolator = Noise_isolation(
        input_file=input_file,
        output_dir=output_dir,
    )

    ndf, fdf = noise_isolator.isolate(type='hard')

    print(ndf.head())
    print(fdf.head())
