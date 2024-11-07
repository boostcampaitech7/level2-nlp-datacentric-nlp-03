import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from cleanlab.filter import find_label_issues
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
from typing import Optional
from datasets import Dataset
from tqdm import tqdm


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

    def load_data(self, train_path_or_data: str, test_path: str):
        """학습 및 테스트 데이터를 로드합니다."""
        try:
            self.train_df = pd.read_csv(train_path_or_data)
            self.test_df = pd.read_csv(test_path)
        except FileNotFoundError as e:
            print(f"파일을 찾을 수 없습니다: {e}")
            raise

        # 학습 및 테스트 데이터 텍스트와 라벨 추출
        self.train_texts = self.train_df['text'].tolist()
        self.train_labels = self.train_df['target'].values
        self.test_texts = self.test_df['text'].tolist()
        self.test_labels = self.test_df['target'].values
        print("데이터 로드가 완료되었습니다.")

    def train_model(self):
        """모델 학습을 수행합니다."""
        dataset = LabeledDataset(self.train_texts, self.train_labels, self.tokenizer)
        
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
        """입력 텍스트에 대한 예측 확률을 계산합니다."""
        if not self.is_trained:
            raise Exception("먼저 모델을 학습시켜야 합니다.")
        
        dataset = LabeledDataset(texts, [0] * len(texts), self.tokenizer)  # 임의의 라벨
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

    def detect_label_errors(self, n_jobs=1):
        """라벨 에러를 검출합니다."""
        print("예측 확률 계산 중...")
        pred_probs = self.predict_proba(self.test_texts)
        
        print("라벨 에러 검출 중...")
        self.label_errors = find_label_issues(
            labels=self.test_labels,
            pred_probs=pred_probs,
            return_indices_ranked_by='self_confidence',
            n_jobs=n_jobs
        )
        print(f"검출된 라벨 에러 개수: {len(self.label_errors)}")
        return self.label_errors

    def correct_labels(self):
        """검출된 라벨 에러를 수정합니다."""
        if self.label_errors is None:
            raise Exception("먼저 라벨 에러를 검출해야 합니다.")
        
        y_corrected = self.test_labels.copy()
        
        # 에러 인덱스에 대해 모델의 예측 라벨로 수정
        error_texts = [self.test_texts[idx] for idx in self.label_errors]
        predicted_labels = np.argmax(self.predict_proba(error_texts), axis=1)
        y_corrected[self.label_errors] = predicted_labels
        print(f"수정된 라벨 에러 개수: {len(self.label_errors)}")
        return y_corrected

    def save_corrected_data(self, output_path="corrected_train_data.csv"):
        """수정된 라벨 데이터를 저장합니다."""
        corrected_labels = self.correct_labels()
        corrected_data = self.test_df.copy()
        corrected_data['target'] = corrected_labels  # 수정된 라벨로 덮어쓰기
        corrected_data.to_csv(output_path, index=False)
        print(f"수정된 데이터가 '{output_path}'에 저장되었습니다.")

    def run(self, train_path_or_data, test_path, output_path="corrected_train_data.csv"):
        """전체 파이프라인을 실행합니다."""
        self.load_data(train_path_or_data, test_path)
        self.train_model()
        self.detect_label_errors()
        self.save_corrected_data(output_path)

class ReLabelingEnsemble:
    def __init__(self, model_names=["klue/bert-base", "klue/roberta-base", "klue/roberta-large", "klue/roberta-small"], train_path_or_data=None, relabeling_path_or_data=None, max_length=128, num_labels=7, seed=456):
        self.model_names = model_names
        self.train_path_or_data = train_path_or_data
        self.relabeling_path_or_data = relabeling_path_or_data
        self.max_length = max_length
        self.num_labels = num_labels
        self.seed = seed
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # 시드 설정
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        # 데이터 로드
        self.noise_data = self.load_data(self.train_path_or_data)
        self.clean_data = self.load_data(self.relabeling_path_or_data)
        
        # 모델 및 토크나이저 로드
        self.models_and_tokenizers = [self.load_model_and_tokenizer(model_name) for model_name in self.model_names]
        
        # Trainer 리스트 초기화
        self.trainers = []

    def load_data(self, data_source):
        """데이터를 로드하거나 DataFrame을 반환합니다."""
        if isinstance(data_source, str):
            # 파일 경로인 경우
            return pd.read_csv(data_source)
        elif isinstance(data_source, pd.DataFrame):
            # DataFrame인 경우
            return data_source
        else:
            raise ValueError("data_source는 파일 경로(str) 또는 pandas DataFrame이어야 합니다.")

    def load_model_and_tokenizer(self, model_name):
        """모델과 토크나이저를 로드합니다."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=self.num_labels).to(self.device)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return model, tokenizer

    def tokenize_function(self, examples, tokenizer):
        """데이터 토크나이즈 함수."""
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=self.max_length)

    def prepare_datasets(self):
        """노이즈 데이터셋과 클린 데이터셋을 준비합니다."""
        # 노이즈 데이터셋 준비
        self.train_dataset = Dataset.from_pandas(self.noise_data[['text', 'target']])
        self.train_dataset = self.train_dataset.map(lambda x: self.tokenize_function(x, self.models_and_tokenizers[0][1]), batched=True)
        self.train_dataset = self.train_dataset.rename_column("target", "labels")

        # 클린 데이터셋 준비
        self.clean_dataset = Dataset.from_dict({"text": self.clean_data['text'].tolist()})
        self.clean_dataset = self.clean_dataset.map(lambda x: self.tokenize_function(x, self.models_and_tokenizers[0][1]), batched=True)

    def train_models(self):
        """모든 모델을 학습합니다."""
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="no",
            learning_rate=2e-5,
            per_device_train_batch_size=32,
            num_train_epochs=3,
            weight_decay=0.01,
            seed=self.seed,
        )
        
        for model, tokenizer in self.models_and_tokenizers:
            tokenized_train_dataset = self.train_dataset.map(lambda x: self.tokenize_function(x, tokenizer), batched=True)
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train_dataset,
            )
            print(f"Training model {model.config._name_or_path}...")
            trainer.train()
            self.trainers.append(trainer)

    def relabel_texts_ensemble(self):
        """앙상블을 사용하여 텍스트 재라벨링을 수행합니다."""
        predictions = []
        for trainer in self.trainers:
            pred = trainer.predict(self.clean_dataset).predictions
            predictions.append(pred)

        # 모든 모델의 예측을 평균하여 최종 예측 생성
        ensemble_predictions = sum(predictions) / len(predictions)
        predicted_labels = ensemble_predictions.argmax(axis=1)
        return predicted_labels

    def relabel_data(self, output_path="relabeled_clean_train.csv"):
        """재라벨링된 데이터를 CSV 파일로 저장합니다."""
        self.clean_data['target'] = self.relabel_texts_ensemble()
        final_data = self.clean_data[['ID', 'text', 'target']]

        # final_data.to_csv(output_path, index=False)
        # print(f"재라벨링된 데이터가 '{output_path}'에 저장되었습니다.")

        return final_data

    def run(self):
        """전체 파이프라인을 실행합니다."""
        self.prepare_datasets()
        self.train_models()
        self.relabel_data()



# 사용 예시
if __name__ == "__main__":

    lec = LabelErrorCorrector(
        model_name='klue/roberta-base', 
        num_labels=7,  # 0~6까지 총 7개 클래스
        batch_size=16,
        epochs=3,
        learning_rate=2e-5,
        random_state=42
    )
    
    # 파이프라인 실행
    lec.run(
        train_path_or_data="/data/ephemeral/home/Yunseo_DCTC/code/last.csv",
        test_path="/data/ephemeral/home/Yunseo_DCTC/data/train.csv",
        output_path="corrected_train_data.csv"
    )

    relabeling_ensemble = ReLabelingEnsemble(
        train_path_or_data="expanded_noise_train.csv",
        relabeling_path_or_data="clean_train.csv",
        max_length=128,
        num_labels=7,
        seed=456
    )
    
    # 파이프라인 실행
    relabeling_ensemble.run()
