import os
import pandas as pd

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
OUTPUT_DIR = os.path.join(BASE_DIR, '../output')

data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

label_mapping = {
    0: '정치',
    1: '경제',
    2: '사회',
    3: '생활문화',
    4: '세계',
    5: 'IT과학',
    6: '스포츠'
}

data['label'] = data['target'].map(label_mapping)

data.to_csv(os.path.join(OUTPUT_DIR, 'train_named.csv'), index=False)