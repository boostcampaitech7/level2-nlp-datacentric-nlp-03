import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm  # 프로그레스 바를 위한 라이브러리

# 1. 데이터 및 출력 디렉토리 설정
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
OUTPUT_DIR = os.path.join(BASE_DIR, '../output')

# 데이터 파일 경로 설정
data_file = os.path.join(DATA_DIR, 'train.csv')

# 출력 디렉토리 설정 (필요 시 생성)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. 데이터 로드
print("데이터 로드 중...")
try:
    data = pd.read_csv(data_file)
    print("데이터 로드 완료.")
except FileNotFoundError:
    print(f"Error: '{data_file}' 파일을 찾을 수 없습니다.")
    exit(1)

# 3. 전처리 함수 정의
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    return text

print("텍스트 전처리 중...")
data['clean_text'] = data['text'].apply(preprocess_text)
print("텍스트 전처리 완료.")

# 4. Sentence Embedding 생성
model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'  # 한국어 지원 모델
print(f"SentenceTransformer 모델 '{model_name}' 로드 중...")
model = SentenceTransformer(model_name)
print("모델 로드 완료.")

print("Sentence Embedding 생성 중...")
embeddings = model.encode(data['clean_text'].tolist(), show_progress_bar=True)
print("Sentence Embedding 생성 완료.")

# 5. 텍스트 품질 평가 모델 사용
print("텍스트 품질 평가를 위한 분류 파이프라인 로드 중...")
classifier = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
print("분류 파이프라인 로드 완료.")

print("텍스트 품질 평가 중...")
quality_scores = classifier(data['clean_text'].tolist())
print("텍스트 품질 평가 완료.")

# 품질 점수 기반 필터링 (1~2 star은 낮은 품질로 간주)
print("품질 점수 기반 필터링 중...")
data['quality_score'] = [score['label'] for score in quality_scores]
data['is_noise_quality'] = data['quality_score'].isin(['1 star', '2 stars'])
print("품질 점수 기반 필터링 완료.")

# 이상치 식별
noise_data_quality = data[data['is_noise_quality']]
clean_data_quality = data[~data['is_noise_quality']]

print(f"Number of noise points detected by quality assessment: {len(noise_data_quality)}")
print(f"Number of clean points: {len(clean_data_quality)}")

# 6. 노이즈 및 클린 데이터 시각화 (t-SNE)
print("t-SNE를 이용한 노이즈 시각화 수행 중...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
embeddings_tsne = tsne.fit_transform(embeddings)
data['tsne-2d-one'] = embeddings_tsne[:,0]
data['tsne-2d-two'] = embeddings_tsne[:,1]
print("t-SNE 수행 완료.")

plt.figure(figsize=(10,8))
sns.scatterplot(
    x='tsne-2d-one', y='tsne-2d-two',
    hue='is_noise_quality',
    palette={False: 'blue', True: 'red'},  # 불리언 키로 수정
    data=data,
    legend="full",
    alpha=0.7
)
plt.title('Noise Detection using Text Quality Assessment and t-SNE')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(title='Is Noise')
plt.tight_layout()
tsne_noise_path = os.path.join(OUTPUT_DIR, "tsne_noise_detection.png")
plt.savefig(tsne_noise_path)
plt.close()

print(f"t-SNE noise detection 그래프가 '{tsne_noise_path}' 파일로 저장되었습니다.")

# 7. 노이즈 및 클린 데이터 시각화 (PCA)
print("PCA를 이용한 노이즈 시각화 수행 중...")
pca = PCA(n_components=2, random_state=42)
embeddings_pca = pca.fit_transform(embeddings)
data['pca-2d-one'] = embeddings_pca[:,0]
data['pca-2d-two'] = embeddings_pca[:,1]
print("PCA 수행 완료.")

plt.figure(figsize=(10,8))
sns.scatterplot(
    x='pca-2d-one', y='pca-2d-two',
    hue='is_noise_quality',
    palette={False: 'green', True: 'orange'},  # 불리언 키로 수정
    data=data,
    legend="full",
    alpha=0.7
)
plt.title('Noise Detection using Text Quality Assessment and PCA')
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.legend(title='Is Noise')
plt.tight_layout()
pca_noise_path = os.path.join(OUTPUT_DIR, "pca_noise_detection.png")
plt.savefig(pca_noise_path)
plt.close()

print(f"PCA noise detection 그래프가 '{pca_noise_path}' 파일로 저장되었습니다.")

# 8. 클러스터별 상위 단어 추출 및 시각화 함수
def get_top_words(texts, top_n=20):
    words = ' '.join(texts).split()
    counter = Counter(words)
    most_common = counter.most_common(top_n)
    return {word: freq for word, freq in most_common}

print("클러스터별 상위 단어 시각화 중...")
for cluster_num in tqdm([True, False], desc="Top Words Visualization"):
    if cluster_num:  # 노이즈인 경우
        cluster_label = "Noise"
        cluster_texts = noise_data_quality['clean_text']
    else:  # 노이즈가 아닌 경우
        cluster_label = "Clean"
        cluster_texts = clean_data_quality['clean_text']
    
    top_freq = get_top_words(cluster_texts, top_n=20)
    
    # 막대 그래프 생성
    plt.figure(figsize=(12,8))
    sns.barplot(x=list(top_freq.values()), y=list(top_freq.keys()), palette="viridis")
    plt.title(f"Top Words in {cluster_label} Data")
    plt.xlabel("Word Frequency")
    plt.ylabel("Word")
    plt.tight_layout()
    
    # 그래프 저장
    top_words_path = os.path.join(OUTPUT_DIR, f"{cluster_label.lower()}_top_words.png")
    plt.savefig(top_words_path)
    plt.close()
    
    print(f"Top words graph for {cluster_label} data가 '{top_words_path}' 파일로 저장되었습니다.")

# 9. 최종 데이터 저장
final_csv_path = os.path.join(OUTPUT_DIR, "classified_data_with_quality_assessment.csv")
data.to_csv(final_csv_path, index=False)
print(f"Clustered and noise-detected data가 '{final_csv_path}' 파일로 저장되었습니다.")
