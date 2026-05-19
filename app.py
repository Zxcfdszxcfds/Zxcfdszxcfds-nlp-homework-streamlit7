import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import Word2Vec, FastText
from gensim.downloader import load
import re

# ---------------------- 页面配置 ----------------------
st.set_page_config(
    page_title="词向量模型对比平台",
    page_icon="📊",
    layout="wide"
)

# ---------------------- 工具函数 ----------------------
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------------------- 模块1：TF-IDF与LSA ----------------------
def module_tfidf_lsa():
    st.header("📊 传统统计模型 (TF-IDF & LSA/A)")
    text = st.text_area("输入英文语料", value="""
The quick brown fox jumps over the lazy dog. 
The dog chases the fox through the forest. 
The fox is quick and clever. 
The dog is loyal and brave.
""")
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if not sentences:
        st.warning("请输入有效语料")
        return
    
    # TF-IDF
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(sentences)
    keywords = tfidf.get_feature_names_out()
    scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    top5_idx = scores.argsort()[-5:][::-1]
    top5_keywords = [(keywords[i], scores[i]) for i in top5_idx]
    
    st.subheader("TF-IDF 关键词Top5")
    st.table(top5_keywords)
    
    # LSA降维可视化
    lsa = TruncatedSVD(n_components=2)
    lsa_vectors = lsa.fit_transform(tfidf_matrix)
    word_vectors = lsa.components_.T
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(word_vectors[:, 0], word_vectors[:, 1], c='blue')
    for i, word in enumerate(keywords):
        ax.annotate(word, (word_vectors[i, 0], word_vectors[i, 1]))
    st.pyplot(fig)

# ---------------------- 模块2：Word2Vec训练与对比 ----------------------
def module_word2vec():
    st.header("🔤 Word2Vec训练与对比 (CBOW vs Skip-Gram)")
    text = st.text_area("输入训练语料", value="""
The quick brown fox jumps over the lazy dog. 
The dog chases the fox through the forest. 
The fox is quick and clever. 
The dog is loyal and brave.
""")
    sentences = [re.findall(r'\b\w+\b', s.lower()) for s in text.split(".") if s.strip()]
    sg = st.radio("训练架构", ["CBOW (sg=0)", "Skip-Gram (sg=1)"])
    sg_val = 1 if sg == "Skip-Gram (sg=1)" else 0
    window = st.slider("上下文窗口", 2, 10, 5)
    
    model = Word2Vec(sentences, sg=sg_val, window=window, vector_size=50, min_count=1)
    target_word = st.text_input("输入查询词", value="fox")
    if target_word in model.wv:
        sim_words = model.wv.most_similar(target_word, topn=5)
        st.subheader(f"与 '{target_word}' 最相似的5个词")
        st.table(sim_words)
    else:
        st.warning("该词未在训练语料中出现")

# ---------------------- 模块3：预训练模型与词类比 (GloVe) ----------------------
def module_glove():
    st.header("🌍 预训练模型与词类比 (GloVe)")
    with st.spinner("加载预训练GloVe模型..."):
        glove = load("glove-twitter-25")
    
    st.subheader("词类比计算器 (A - B + C)")
    col1, col2, col3 = st.columns(3)
    with col1:
        a = st.text_input("A", value="king")
    with col2:
        b = st.text_input("B", value="man")
    with col3:
        c = st.text_input("C", value="woman")
    
    if a in glove and b in glove and c in glove:
        result_vec = glove[a] - glove[b] + glove[c]
        result = glove.most_similar([result_vec], topn=1)[0]
        st.success(f"类比结果: {a} - {b} + {c} ≈ {result[0]} (相似度: {result[1]:.4f})")
    else:
        st.warning("部分词不在预训练词表中")
    
    st.subheader("词义相似度计算")
    w1 = st.text_input("单词1", value="king")
    w2 = st.text_input("单词2", value="queen")
    if w1 in glove and w2 in glove:
        sim = cosine_sim(glove[w1], glove[w2])
        st.info(f"相似度: {sim:.4f}")

# ---------------------- 模块4：子词特征与句向量 (FastText & Sent2Vec) ----------------------
def module_fasttext_sent2vec():
    st.header("⚡ FastText与句向量表示")
    text = st.text_area("输入训练语料", value="""
The quick brown fox jumps over the lazy dog. 
The dog chases the fox through the forest. 
The fox is quick and clever. 
The dog is loyal and brave.
""")
    sentences = [re.findall(r'\b\w+\b', s.lower()) for s in text.split(".") if s.strip()]
    ft_model = FastText(sentences, vector_size=50, window=5, min_count=1)
    
    st.subheader("OOV测试 (FastText vs Word2Vec)")
    oov_word = st.text_input("输入带拼写错误的词", value="computeer")
    col1, col2 = st.columns(2)
    with col1:
        try:
            w2v_model = Word2Vec(sentences, vector_size=50, window=5, min_count=1)
            vec = w2v_model.wv[oov_word]
            st.success(f"Word2Vec: 存在向量")
        except KeyError:
            st.error("Word2Vec: 未登录词 (KeyError)")
    with col2:
        ft_vec = ft_model.wv[oov_word]
        similar = ft_model.wv.most_similar(oov_word, topn=3)
        st.success(f"FastText: 成功处理，相似词: {[w[0] for w in similar]}")
    
    st.subheader("句向量相似度 (Sent2Vec简化版)")
    s1 = st.text_input("句子1", value="The quick brown fox jumps.")
    s2 = st.text_input("句子2", value="The fast brown fox leaps.")
    
    def get_sent_vec(sent, model):
        words = re.findall(r'\b\w+\b', sent.lower())
        vecs = [model.wv[w] for w in words if w in model.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)
    
    v1 = get_sent_vec(s1, ft_model)
    v2 = get_sent_vec(s2, ft_model)
    sim = cosine_sim(v1, v2)
    st.info(f"句向量余弦相似度: {sim:.4f}")

# ---------------------- 主界面 ----------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "模块1: TF-IDF & LSA",
    "模块2: Word2Vec",
    "模块3: GloVe词类比",
    "模块4: FastText & Sent2Vec"
])

with tab1:
    module_tfidf_lsa()
with tab2:
    module_word2vec()
with tab3:
    module_glove()
with tab4:
    module_fasttext_sent2vec()

st.markdown("---")
st.markdown("© 2025 NLP 课程实验 | 词向量模型对比平台")
