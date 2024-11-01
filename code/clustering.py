import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm  # tqdm 추가

# 현재 작업 디렉토리 설정
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, '../data')
OUTPUT_DIR = os.path.join(BASE_DIR, '../output')

# 데이터 로드
data = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

# 그래프를 저장할 디렉토리 설정
output_dir = "/data/ephemeral/home/Yunseo_DCTC/output"

# 디렉토리가 존재하지 않으면 생성
os.makedirs(output_dir, exist_ok=True)

# 2. 전처리 함수 정의
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    return text

print("Preprocessing text...")
data['clean_text'] = data['text'].apply(preprocess_text)
print("Text preprocessing completed.")

# 3. Sentence Embedding 생성 (멀티링궐 모델 사용)
model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'  # 한국어 지원 모델
print(f"Loading model '{model_name}'...")
model = SentenceTransformer(model_name)
print("Model loaded.")

print("Generating sentence embeddings...")
embeddings = model.encode(data['clean_text'].tolist(), show_progress_bar=True)
print("Sentence embeddings generated.")

# 4. 클러스터 수 결정 (엘보우 방법 - KMeans)
print("Determining the optimal number of clusters using the Elbow Method...")
inertia = []
K = range(2, 10)

for k in tqdm(K, desc="Elbow Method Progress"):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(embeddings)
    inertia.append(kmeans.inertia_)

# 엘보우 그래프 그리기 및 저장
plt.figure(figsize=(8,6))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.xticks(K)
plt.tight_layout()
elbow_path = os.path.join(output_dir, "elbow_method.png")
plt.savefig(elbow_path)
plt.close()

print(f"Elbow graph saved at '{elbow_path}'.")

# 5. 최적의 클러스터 수 설정 (예: k=2)
optimal_k = 2  # Based on the Elbow Method graph
print(f"Optimal number of clusters selected: k={optimal_k}")

# 6. K-Means 클러스터링 수행
print(f"Performing K-Means clustering with k={optimal_k}...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(embeddings)
data['cluster'] = clusters
print("K-Means clustering completed.")

# 클러스터 분포 시각화 및 저장
plt.figure(figsize=(8,6))
sns.countplot(x='cluster', data=data)
plt.title("Cluster Distribution")
plt.xlabel("Cluster")
plt.ylabel("Number of Data Points")
plt.tight_layout()
cluster_dist_path = os.path.join(output_dir, "cluster_distribution.png")
plt.savefig(cluster_dist_path)
plt.close()

print(f"Cluster distribution graph saved at '{cluster_dist_path}'.")

# 7. DBSCAN을 사용한 노이즈 검출
print("Performing DBSCAN clustering for noise detection...")
# DBSCAN 파라미터 설정 (필요에 따라 조정)
dbscan_eps = 0.5
dbscan_min_samples = 5

dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric='euclidean')
dbscan_clusters = dbscan.fit_predict(embeddings)
data['dbscan_cluster'] = dbscan_clusters

# DBSCAN 클러스터 분포 시각화 및 저장
plt.figure(figsize=(8,6))
sns.countplot(x='dbscan_cluster', data=data)
plt.title("DBSCAN Cluster Distribution")
plt.xlabel("Cluster")
plt.ylabel("Number of Data Points")
plt.tight_layout()
dbscan_cluster_dist_path = os.path.join(output_dir, "dbscan_cluster_distribution.png")
plt.savefig(dbscan_cluster_dist_path)
plt.close()

print(f"DBSCAN cluster distribution graph saved at '{dbscan_cluster_dist_path}'.")

# DBSCAN 노이즈 포인트 확인
noise_points = data[data['dbscan_cluster'] == -1]
print(f"Number of noise points detected by DBSCAN: {len(noise_points)}")

# 8. t-SNE 시각화 (DBSCAN 결과 포함)
print("Performing t-SNE visualization for DBSCAN clusters...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
embeddings_tsne = tsne.fit_transform(embeddings)
data['tsne-2d-one'] = embeddings_tsne[:,0]
data['tsne-2d-two'] = embeddings_tsne[:,1]

plt.figure(figsize=(10,8))
sns.scatterplot(
    x='tsne-2d-one', y='tsne-2d-two',
    hue='dbscan_cluster',
    palette=sns.color_palette("hsv", len(set(dbscan_clusters))),
    data=data,
    legend="full",
    alpha=0.7
)
plt.title('DBSCAN Cluster Visualization using t-SNE')
plt.tight_layout()
tsne_path = os.path.join(output_dir, "dbscan_tsne_clustering.png")
plt.savefig(tsne_path)
plt.close()

print(f"DBSCAN t-SNE cluster visualization graph saved at '{tsne_path}'.")

# 9. PCA 시각화 (DBSCAN 결과 포함)
print("Performing PCA visualization for DBSCAN clusters...")
pca = PCA(n_components=2, random_state=42)
embeddings_pca = pca.fit_transform(embeddings)
data['pca-2d-one'] = embeddings_pca[:,0]
data['pca-2d-two'] = embeddings_pca[:,1]

plt.figure(figsize=(10,8))
sns.scatterplot(
    x='pca-2d-one', y='pca-2d-two',
    hue='dbscan_cluster',
    palette=sns.color_palette("hsv", len(set(dbscan_clusters))),
    data=data,
    legend="full",
    alpha=0.7
)
plt.title('DBSCAN Cluster Visualization using PCA')
plt.tight_layout()
pca_path = os.path.join(output_dir, "dbscan_pca_clustering.png")
plt.savefig(pca_path)
plt.close()

print(f"DBSCAN PCA cluster visualization graph saved at '{pca_path}'.")

# 10. 클러스터별 상위 단어 추출 및 시각화 함수
def get_top_words(texts, top_n=20):
    words = ' '.join(texts).split()
    counter = Counter(words)
    most_common = counter.most_common(top_n)
    return {word: freq for word, freq in most_common}

# 클러스터별 상위 단어 시각화 및 저장 (DBSCAN 결과)
print("Visualizing top words for each DBSCAN cluster...")
for cluster_num in tqdm(sorted(set(dbscan_clusters)), desc="Top Words Visualization"):
    if cluster_num == -1:
        # 노이즈 클러스터는 건너뜀
        continue
    cluster_texts = data[data['dbscan_cluster'] == cluster_num]['clean_text']
    top_freq = get_top_words(cluster_texts, top_n=20)
    
    # 막대 그래프 생성
    plt.figure(figsize=(12,8))
    sns.barplot(x=list(top_freq.values()), y=list(top_freq.keys()), palette="viridis")
    plt.title(f"Top Words in DBSCAN Cluster {cluster_num}")
    plt.xlabel("Word Frequency")
    plt.ylabel("Word")
    plt.tight_layout()
    
    # 그래프 저장
    top_words_path = os.path.join(output_dir, f"dbscan_cluster_{cluster_num}_top_words.png")
    plt.savefig(top_words_path)
    plt.close()
    
    print(f"Top words graph for DBSCAN Cluster {cluster_num} saved at '{top_words_path}'.")

# 11. 최종 데이터 저장
final_csv_path = os.path.join(output_dir, "classified_data_with_dbscan.csv")
data.to_csv(final_csv_path, index=False)
print(f"Clustered data with DBSCAN saved at '{final_csv_path}'.")
