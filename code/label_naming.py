import os
import pandas as pd

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
OUTPUT_DIR = os.path.join(BASE_DIR, '../output')

data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

label_mapping = {
    0: '생활문화',
    1: '스포츠',
    2: '정치',
    3: '사회',
    4: 'IT과학',
    5: '경제',
    6: '세계',
}

data['label'] = data['target'].map(label_mapping)

data.to_csv(os.path.join(OUTPUT_DIR, 'train_named.csv'), index=False)