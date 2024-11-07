from flask import Flask, render_template, request, send_file, jsonify
import requests
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from wordcloud import WordCloud
import io
import nltk
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time

# 確保 NLTK 資料下載
nltk_data_path = './nltk_data'
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

try:
    nltk.download('stopwords', download_dir=nltk_data_path)
    nltk.download('punkt', download_dir=nltk_data_path)
except Exception as e:
    print(f"下載 NLTK 資料時出錯: {e}")

# 將 NLTK 資料路徑添加到系統路徑
nltk.data.path.append(nltk_data_path)

# 停用詞與標點符號移除
stop_words = set(stopwords.words('english'))
punctuation_table = str.maketrans('', '', string.punctuation)

app = Flask(__name__)

# PubMed API 基本 URL
PUBMED_API_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

# PubMed API 搜尋函數
def fetch_pubmed_articles(keyword, count):
    params = {
        "db": "pubmed",
        "term": keyword,
        "retmax": count,
        "retmode": "json",
        "usehistory": "y"
    }
    response = requests.get(PUBMED_API_URL, params=params)
    if response.status_code != 200:
        print(f"API 請求失敗: {response.status_code}")
        return []  # 處理請求失敗的情況

    article_ids = response.json().get("esearchresult", {}).get("idlist", [])
    if not article_ids:
        print("未找到文章")
        return []  # 處理未找到文章的情況

    preprocessed_summaries = []
    batch_size = 50  # 減少每批請求的大小
    for i in range(0, len(article_ids), batch_size):
        batch_ids = article_ids[i:i + batch_size]
        summary_params = {
            "db": "pubmed",
            "id": ",".join(batch_ids),
            "retmode": "json"
        }
        summary_response = requests.get(PUBMED_SUMMARY_URL, params=summary_params)
        if summary_response.status_code != 200:
            print(f"摘要請求失敗: {summary_response.status_code}")
            continue  # 如果這批請求失敗，則跳過

        articles = summary_response.json().get("result", {})
        for article_id in batch_ids:
            article = articles.get(article_id, {})
            title = article.get("title", "")
            if title:
                preprocessed_summaries.append(preprocess_text(title))
        
        time.sleep(1)  # 在每批請求之間添加延遲

    return preprocessed_summaries

# 預處理文本
def preprocess_text(text):
    words = word_tokenize(text.lower().translate(punctuation_table))
    filtered_words = [w for w in words if w.isalpha() and w not in stop_words]
    return filtered_words

# 構建文字雲
def generate_word_cloud(text):
    wordcloud = WordCloud(width=850, height=500, background_color='white').generate(text)
    image_stream = io.BytesIO()
    wordcloud.to_image().save(image_stream, 'PNG')
    image_stream.seek(0)
    return image_stream

# 構建 PCA 圖
def generate_pca_plot(model):
    words = list(model.wv.index_to_key)
    word_vectors = model.wv[words]

    pca = PCA(n_components=2)
    word_pca = pca.fit_transform(word_vectors)

    plt.figure(figsize=(10, 10))
    plt.scatter(word_pca[:, 0], word_pca[:, 1])

    for i, word in enumerate(words):
        plt.annotate(word, xy=(word_pca[i, 0], word_pca[i, 1]))

    image_stream = io.BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    return image_stream

# 路由處理首頁請求
@app.route('/')
def index():
    return render_template('index.html')

# 搜尋並生成文字雲
@app.route('/search', methods=['POST'])
def search():
    data = request.json
    keyword = data.get("keyword")
    count = data.get("count")

    print(f"Keyword: {keyword}, Count: {count}")  # 輸出關鍵字和數量到控制台

    preprocessed_text = fetch_pubmed_articles(keyword, count)

    if not preprocessed_text:
        return jsonify({"error": "未找到任何文章"}), 404

    # 訓練 Word2Vec 模型
    cbow_model = Word2Vec(sentences=preprocessed_text, vector_size=100, window=5, min_count=1, sg=0)

    # 把詞按頻率拼接成一個字串，用於生成文字雲
    all_words = ' '.join([' '.join(words) for words in preprocessed_text])
    word_cloud_img = generate_word_cloud(all_words)
    
    return send_file(word_cloud_img, mimetype='image/png')

# 生成 PCA 圖
@app.route('/pca', methods=['POST'])
def pca():
    data = request.json
    keyword = data.get("keyword")
    count = data.get("count")

    preprocessed_text = fetch_pubmed_articles(keyword, count)

    if not preprocessed_text:
        return jsonify({"error": "未找到任何文章"}), 404

    # 訓練 Word2Vec 模型
    cbow_model = Word2Vec(sentences=preprocessed_text, vector_size=100, window=5, min_count=1, sg=0)

    pca_plot_img = generate_pca_plot(cbow_model)
    
    return send_file(pca_plot_img, mimetype='image/png')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # 預設端口設為 5000
    app.run(host='0.0.0.0', port=port, debug=True)