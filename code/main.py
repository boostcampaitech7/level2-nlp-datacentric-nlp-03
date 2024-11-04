from noise_isolation import Noise_isolation
from augmentation import Augmentation
import os

# 입력 및 출력 파일 경로 설정
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
OUTPUT_DIR = os.path.join(BASE_DIR, '../output')

input_file = os.path.join(DATA_DIR, 'train.csv')
noise_isolated_output_file = os.path.join(OUTPUT_DIR, 'noise_isolated.csv')
augmentation_output_file = os.path.join(OUTPUT_DIR, 'augmented')

# 허깅 페이스 토큰 설정
hf_token = "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"

# 클래스 인스턴스 생성
korean_text_filter = Noise_isolation(
    input_file=input_file,
    output_file=noise_isolated_output_file,
    non_korean_ratio_lower_threshold=0.1,  # 필요에 따라 값 조정
    non_korean_ratio_upper_threshold=0.3   # 필요에 따라 값 조정
)

# 프로세스 실행
filltered_df = korean_text_filter.run(save=False)

augmentation = Augmentation(
    input_data=filltered_df,
    output_dir=augmentation_output_file
)

korean_text_filter.run()
augmentation.run_all(hf_token=hf_token)