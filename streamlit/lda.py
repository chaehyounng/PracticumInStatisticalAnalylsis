import pyLDAvis.gensim_models
import pyLDAvis
import gensim
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
import streamlit as st
import pandas as pd
import os

def train_lda_and_visualize(news_df):
    """한 달 치 뉴스 데이터를 바탕으로 LDA 모델 학습 및 pyLDAvis 변환"""
    
    # 'token_lst' 컬럼 활용
    tokenized_documents = news_df["token_lst"].tolist()

    # Gensim Dictionary & Corpus 생성
    dictionary = Dictionary(tokenized_documents)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_documents]

    # LDA 모델 학습
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=3,  # 토픽 개수 = 3
        iterations=100,
        passes=10,
        random_state=0
    )

    # pyLDAvis 시각화 생성
    viz = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary, sort_topics=False)
    html_viz = pyLDAvis.prepared_data_to_html(viz)

    return html_viz
