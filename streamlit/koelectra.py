# koelectra.py
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 모델 및 토크나이저 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_model")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# KoELECTRA 감성 분석 함수
def sentiment_analysis_koelectra(text):
    """Fine-Tuned KoELECTRA 모델을 활용한 감성 분석"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    with torch.no_grad():  # Inference 모드
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()  # 0: 부정, 1: 중립, 2: 긍정

    # KoELECTRA 레이블 변환 
    label_mapping = {0: -1, 1: 1, 2: 0}
    return label_mapping[predicted_class]
