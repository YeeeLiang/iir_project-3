from flask import Flask, render_template, request, jsonify, send_file
import requests
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import nltk
import os

nltk.download('stopwords')

app = Flask(__name__)

# PubMed API base URL
PUBMED_API_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

# 停用詞與標點符號移除
stop_words = set(stopwords.words('english'))
punctuation_table = str.maketrans('', '', string.punctuation)

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
    article_ids = response.json().get("esearchresult", {}).get("idlist", [])

    # 抓取文章摘要並預處理
    preprocessed_summaries = []
    batch_size = 100  # 可以根據需要調整批次大小
    for i in range(0, len(article_ids), batch_size):
        batch_ids = article_ids[i:i+batch_size]
        summary_params = {
            "db": "pubmed",
            "id": ",".join(batch_ids),
            "retmode": "json"
        }
        summary_response = requests.get(PUBMED_SUMMARY_URL, params=summary_params)
        articles = summary_response.json().get("result", {})
        for article_id in article_ids:
            article = articles.get(article_id, {})
            title = article.get("title", "")
            if title:  # 確保有標題
                preprocessed_summaries.append(preprocess_text(title))
    return preprocessed_summaries

# 預處理文本
def preprocess_text(text):
    words = word_tokenize(text.lower().translate(punctuation_table))
    filtered_words = [w for w in words if w.isalpha() and w not in stop_words]
    return filtered_words

# 構建文字雲
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    image_stream = io.BytesIO()
    wordcloud.to_image().save(image_stream, 'PNG')
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
    preprocessed_text = fetch_pubmed_articles(keyword, count)
    
    # 訓練 Word2Vec 模型
    cbow_model = Word2Vec(sentences=preprocessed_text, vector_size=100, window=5, min_count=1, sg=0)

    # 把詞按頻率拼接成一個字串，用於生成文字雲
    all_words = ' '.join([' '.join(words) for words in preprocessed_text])
    word_cloud_img = generate_word_cloud(all_words)
    
    return send_file(word_cloud_img, mimetype='image/png')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # 預設端口設為 5000
    app.run(host='0.0.0.0', port=port, debug=True)
