import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm

def predict_sales(df):
    """판매량 예측 및 그래프 출력"""

    # year_month 변환 (CSV에서 "2023-01" 형식으로 들어온 데이터 처리)
    df["year_month"] = pd.to_datetime(df["year_month"], format="%Y-%m")
    df["year"] = df["year_month"].dt.year
    df["month"] = df["year_month"].dt.month

    # 로그 변환 (모델 학습용)
    df["sales"] = np.log1p(df["sales"])
    df["previous_month_sales"] = np.log1p(df["previous_month_sales"])

    # train & test 분리
    train = df.dropna(subset=["sales"])  # 실제 판매량 있는 데이터 (학습용)
    test = df[df["sales"].isna()]  # 판매량이 비어있는 데이터 (예측 대상)

    # OLS 회귀 모델 학습
    model_fit = smf.ols("sales ~ pnr_naver + previous_month_sales - 1", data=train).fit()

    # 예측 수행
    pred_y = model_fit.predict(test)

    # 예측값 저장
    test["predicted_sales"] = np.expm1(pred_y).fillna(0)

    # 예측된 판매량 출력
    st.subheader("📈 예측된 판매량")
    st.dataframe(test[["year_month", "predicted_sales"]])

    # 한글 폰트 설정 (NanumGothic)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(BASE_DIR, "fonts", "NanumGothic.ttf")
    font_prop = fm.FontProperties(fname=font_path)

    # 그래프 출력
    st.subheader("판매량 추이 및 예측")
    fig, ax = plt.subplots(figsize=(10, 5))

    # 실제 판매량 (파란색)
    past_data = df.dropna(subset=["sales"])
    ax.plot(past_data["year_month"], np.expm1(past_data["sales"]),
            marker="o", linestyle="-", color="blue", label="실제 판매량")

    # 예측 판매량 (빨간색 점선)
    if not test.empty:
        ax.plot(test["year_month"], test["predicted_sales"],
                marker="o", linestyle="dashed", color="red", label="예측 판매량")

    ax.set_xlabel("날짜", fontproperties=font_prop)
    ax.set_ylabel("판매량", fontproperties=font_prop)
    ax.set_title("🚗 판매량 추이 및 예측", fontproperties=font_prop)
    ax.legend(prop=font_prop)

    st.pyplot(fig)

    # 예측된 판매량 CSV 다운로드 기능 추가
    csv = test[["year_month", "predicted_sales"]].to_csv(index=False, encoding="utf-8-sig")
    st.download_button("📥 예측 결과 CSV 다운로드", data=csv, file_name="predicted_sales.csv", mime="text/csv")
