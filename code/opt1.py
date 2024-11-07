import os
import pandas as pd
from noise_isolation import Noise_isolation
from augmentation import Augmentation
from relabeling import ReLabelingEnsemble

# 입력 및 출력 파일 경로 설정
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
input_file = os.path.join(DATA_DIR, 'small.csv')

noise_isolator = Noise_isolation(
    input_file=input_file,
    output_dir=DATA_DIR
)

augmentor = Augmentation(
    output_dir=DATA_DIR
)

# isolation 실행
noised_df, filtered_df = noise_isolator.isolate(type='hard')

print(noised_df)
print(filtered_df)

# 노이지 데이터 복원 및 증강 
augmented_noised_df = augmentor.naive_augment(noised_df)

print(augmented_noised_df)

# 리라벨링
relabeling_ensemble = ReLabelingEnsemble(
    train_path_or_data=augmented_noised_df,
    relabeling_path_or_data=filtered_df,
)
relabeled_filtered_df = relabeling_ensemble.run()


# # 라벨링 된 filtered_df 증강
# augmented_relabeled_filtered_df = augmentor.easy_data_augmentation(relabeled_filtered_df)
# augmented_relabeled_filtered_df = augmentor.back_translation(relabeled_filtered_df)


# # augmented_relabeled_filtered_df + augmented_noised_df
# last = pd.concat([augmented_noised_df, augmented_relabeled_filtered_df], ignore_index=True)
# print(last.head())

# last.to_csv("last.csv", index=False)