import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from cleanlab.filter import find_label_issues
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from typing import Optional


class LabeledDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class LabelErrorCorrector:
    def __init__(
        self,
        model_name: str = 'klue/roberta-base',
        num_labels: int = 7,
        batch_size: int = 16,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        random_state: int = 42,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 토크나이저 및 모델 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels
        ).to(self.device)
        
        self.is_trained = False
        self.label_errors = None
        self.trainer = None

    def train_model(self, texts, labels):
        dataset = LabeledDataset(texts, labels, self.tokenizer)
        
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            evaluation_strategy='no',
            save_strategy='no',
            logging_dir='./logs',
            logging_steps=10,
            seed=self.random_state
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset
        )
        
        print("모델 학습 시작...")
        self.trainer.train()
        print("모델 학습 완료.")
        self.is_trained = True

    def predict_proba(self, texts):
        if not self.is_trained:
            raise Exception("먼저 모델을 학습시켜야 합니다.")
        
        dataset = LabeledDataset(texts, [0]*len(texts), self.tokenizer)  # 라벨은 임의로 설정
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        
        self.model.eval()
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = {key: val.to(self.device) for key, val in batch.items() if key != 'labels'}
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                all_probs.append(probs.cpu().numpy())
        
        return np.concatenate(all_probs, axis=0)

    def detect_label_errors(self, texts, labels, n_jobs=1):
        print("예측 확률 계산 중...")
        pred_probs = self.predict_proba(texts)
        
        print("라벨 에러 검출 중...")
        label_errors = find_label_issues(
            labels=labels,
            pred_probs=pred_probs,
            return_indices_ranked_by='self_confidence',
            n_jobs=n_jobs
        )
        self.label_errors = label_errors
        print(f"검출된 라벨 에러 개수: {len(label_errors)}")
        return label_errors

    def correct_labels(self, texts, labels):
        if self.label_errors is None:
            raise Exception("먼저 라벨 에러를 검출해야 합니다.")
        
        y_corrected = labels.copy()
        
        # 에러 인덱스에 대해 모델의 예측 라벨로 수정
        error_texts = [texts[idx] for idx in self.label_errors]
        predicted_labels = np.argmax(self.predict_proba(error_texts), axis=1)
        y_corrected[self.label_errors] = predicted_labels
        print(f"수정된 라벨 에러 개수: {len(self.label_errors)}")
        return y_corrected


# 사용 예제
if __name__ == "__main__":
    import sys

    # 학습용 데이터 로드
    try:
        train_df = pd.read_csv("/data/ephemeral/home/Yunseo_DCTC/code/last.csv")
    except FileNotFoundError:
        print("학습용 CSV 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        sys.exit(1)

    # 테스트용 데이터 로드
    try:
        test_df = pd.read_csv("/data/ephemeral/home/Yunseo_DCTC/data/train.csv")
    except FileNotFoundError:
        print("테스트용 CSV 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        sys.exit(1)
    
    # 학습 데이터 확인
    print("학습 데이터 프레임의 처음 몇 줄:")
    print(train_df.head())

    # 테스트 데이터 확인
    print("테스트 데이터 프레임의 처음 몇 줄:")
    print(test_df.head())
    
    # 학습 데이터에서 텍스트와 라벨 추출
    train_texts = train_df['text'].tolist()
    train_labels = train_df['target'].values
    
    # 테스트 데이터에서 텍스트와 라벨, 인덱스 추출
    test_texts = test_df['text'].tolist()
    test_labels = test_df['target'].values
    test_indices = test_df.index.tolist()
    
    # LabelErrorCorrector 클래스 초기화
    lec = LabelErrorCorrector(
        model_name='klue/roberta-base',  # 한국어 RoBERTa 모델
        num_labels=7,  # 0~6까지 총 7개 클래스
        batch_size=16,  # 작은 데이터셋이므로 배치 사이즈를 줄임
        epochs=3,
        learning_rate=2e-5,
        random_state=42
    )
    
    # 학습용 데이터로 모델 학습
    lec.train_model(train_texts, train_labels)
    
    # 테스트용 데이터에서 라벨 에러 검출 및 수정
    label_errors = lec.detect_label_errors(test_texts, test_labels)
    y_corrected = lec.correct_labels(test_texts, test_labels)

    # 수정되지 않은 데이터와 수정된 데이터 결합
    corrected_data = test_df.copy()
    corrected_data['corrected_target'] = y_corrected
    corrected_data['corrected'] = np.where(test_labels == y_corrected, 'No', 'Yes')
    
    # 최종 CSV로 저장 (ID, text, target 형식으로)
    final_data = corrected_data[['ID', 'text', 'target']].copy()
    final_data['target'] = corrected_data['corrected_target']  # 수정된 라벨로 덮어쓰기
    final_data.to_csv("corrected_train_data.csv", index=False)

    print("수정된 데이터가 'corrected_test_data.csv'에 저장되었습니다.")
    
    # 수정된 라벨 확인
    corrected_indices = np.where(y_corrected != test_labels)[0]
    print(f"수정된 라벨 개수: {len(corrected_indices)}")
    print(f"수정된 라벨 인덱스: {corrected_indices}")
    
    # 수정된 라벨 확인
    # for idx in corrected_indices:
    #     original_idx = test_indices[idx]  # 원본 데이터프레임의 인덱스
    #     print(f"ID: {test_df.loc[original_idx, 'ID']}")
    #     print(f"원본 텍스트: {test_df.loc[original_idx, 'text']}")
    #     print(f"원본 라벨: {test_labels[idx]}")
    #     print(f"수정된 라벨: {y_corrected[idx]}")
    #     print("-" * 50)

    # from cleanlab.dataset import health_summary
    # class_names=[0,1,2,3,4,5,6]
    # health_summary(dataset_train['target'], train_pred_probs, class_names=class_names)