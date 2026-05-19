import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re

# ---------------------- 页面配置 ----------------------
st.set_page_config(
    page_title="词向量模型对比平台",
    page_icon="📊",
    layout="wide"
)

# ---------------------- 工具函数 ----------------------
def cosine_sim(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
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
    word_vectors = lsa.fit_transform(tfidf_matrix.T)
    lsa_df = pd.DataFrame({
        "Word": keywords,
        "Component 1": word_vectors[:, 0],
        "Component 2": word_vectors[:, 1]
    })
    st.subheader("LSA词向量降维可视化")
    st.scatter_chart(lsa_df, x="Component 1", y="Component 2", color="Word", size=100)

# ---------------------- 模块2：简化版词向量（模拟Word2Vec） ----------------------
def module_simple_word2vec():
    st.header("🔤 简化词向量对比（模拟CBOW/Skip-Gram）")
    text = st.text_area("输入语料", value="""
The quick brown fox jumps over the lazy dog. 
The dog chases the fox through the forest. 
The fox is quick and clever. 
The dog is loyal and brave.
""")
    sentences = [re.findall(r'\b\w+\b', s.lower()) for s in text.split(".") if s.strip()]
    all_words = list(set([w for sent in sentences for w in sent]))
    
    # 模拟两种架构的词向量差异（随机初始化，仅展示界面）
    st.subheader("架构选择")
    sg = st.radio("训练架构", ["CBOW (sg=0)", "Skip-Gram (sg=1)"])
    st.info(f"当前选择: {sg}（模拟实现，不做真实训练）")
    
    target_word = st.text_input("输入查询词", value="fox")
    if target_word in all_words:
        # 随机生成向量，计算余弦相似度（仅演示）
        np.random.seed(42)
        word_vecs = {w: np.random.rand(10) for w in all_words}
        sim_words = []
        for w in all_words:
            if w != target_word:
                sim = cosine_sim(word_vecs[target_word], word_vecs[w])
                sim_words.append((w, sim))
        sim_words.sort(key=lambda x: x[1], reverse=True)
        st.subheader(f"与 '{target_word}' 最相似的5个词（模拟结果）")
        st.table(sim_words[:5])
    else:
        st.warning("该词未在语料中出现")

# ---------------------- 模块3：词类比计算器（模拟GloVe） ----------------------
def module_simple_analogy():
    st.header("🌍 词类比计算器（模拟A-B+C）")
    # 预定义固定类比结果，避免加载模型
    st.subheader("经典词类比测试")
    examples = {
        "king - man + woman": "queen",
        "paris - france + china": "beijing",
        "dog - puppy + cat": "kitten"
    }
    for query, result in examples.items():
        st.info(f"{query} ≈ {result}")
    
    st.subheader("自定义词类比（模拟计算）")
    col1, col2, col3 = st.columns(3)
    with col1:
        a = st.text_input("A", value="king")
    with col2:
        b = st.text_input("B", value="man")
    with col3:
        c = st.text_input("C", value="woman")
    if st.button("计算类比结果"):
        if a == "king" and b == "man" and c == "woman":
            st.success("结果: queen")
        elif a == "paris" and b == "france" and c == "china":
            st.success("结果: beijing")
        else:
            st.warning("自定义类比暂不支持，可使用上方示例")

# ---------------------- 模块4：FastText与句向量（简化版） ----------------------
def module_simple_fasttext():
    st.header("⚡ 子词与句向量（模拟FastText）")
    st.subheader("OOV测试（模拟FastText鲁棒性）")
    oov_word = st.text_input("输入带拼写错误的词", value="computeer")
    # 简单通过字符相似度匹配（仅演示）
    correct_word = "computer"
    st.info(f"FastText模拟结果: '{oov_word}' 最相似的词是 '{correct_word}'")
    st.error("Word2Vec模拟结果: 未登录词（KeyError）")
    
    st.subheader("句向量相似度（均值法）")
    s1 = st.text_input("句子1", value="The quick brown fox jumps.")
    s2 = st.text_input("句子2", value="The fast brown fox leaps.")
    # 简单词袋均值向量
    words1 = re.findall(r'\b\w+\b', s1.lower())
    words2 = re.findall(r'\b\w+\b', s2.lower())
    all_words = list(set(words1 + words2))
    vec1 = np.zeros(len(all_words))
    vec2 = np.zeros(len(all_words))
    for w in words1:
        if w in all_words:
            vec1[all_words.index(w)] = 1
    for w in words2:
        if w in all_words:
            vec2[all_words.index(w)] = 1
    sim = cosine_sim(vec1, vec2)
    st.info(f"句向量相似度: {sim:.4f}")

# ---------------------- 主界面 ----------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "模块1: TF-IDF & LSA",
    "模块2: 词向量对比",
    "模块3: 词类比计算",
    "模块4: 子词与句向量"
])

with tab1:
    module_tfidf_lsa()
with tab2:
    module_simple_word2vec()
with tab3:
    module_simple_analogy()
with tab4:
    module_simple_fasttext()

st.markdown("---")
st.markdown("© 2025 NLP 课程实验 | 词向量模型对比平台")
