
import pandas as pd
import random
from collections import defaultdict
from koeda import EDA

data_path = '/data/ephemeral/home/Yunseo_DCTC/data/noise_isolated.csv'
data = pd.read_csv(data_path)

random.seed(42)

eda = EDA(morpheme_analyzer="Okt", alpha_sr=0.3, alpha_ri=0.3, alpha_rs=0.3, prob_rd=0.3)

# 데이터셋을 5배로 증강 (원본 + SR + RI + RS + RD)
expanded_data = defaultdict(list)

for idx, row in data.iterrows():
    text = row['text']
    target = row['target']
    record_id = row['ID']

    # Original
    expanded_data['ID'].append(record_id)
    expanded_data['text'].append(text)
    expanded_data['target'].append(target)

    # Synonym Replacement (SR)
    sr_augmented = eda(text, p=(0.1, 0, 0, 0))
    expanded_data['ID'].append(record_id)
    expanded_data['text'].append(sr_augmented)
    expanded_data['target'].append(target)

    # Random Insertion (RI)
    # ri_augmented = eda(text, p=(0, 0.1, 0, 0))
    # expanded_data['ID'].append(record_id)
    # expanded_data['text'].append(ri_augmented)
    # expanded_data['target'].append(target)

    # Random Swap (RS)
    rs_augmented = eda(text, p=(0, 0, 0.1, 0))
    expanded_data['ID'].append(record_id)
    expanded_data['text'].append(rs_augmented)
    expanded_data['target'].append(target)

    # Random Deletion (RD)
    rd_augmented = eda(text, p=(0, 0, 0, 0.1))
    expanded_data['ID'].append(record_id)
    expanded_data['text'].append(rd_augmented)
    expanded_data['target'].append(target)

# DataFrame으로 변환하고 저장
expanded_df = pd.DataFrame(expanded_data)
expanded_data_path = 'expanded_noise_and_relabeled_clean_train.csv'
expanded_df.to_csv(expanded_data_path, index=False)