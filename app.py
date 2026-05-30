
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 지도학습 데이터 준비 (기대수명과 BMI, GDP, Alcohol)
url = "https://github.com/dongupak/DataML/raw/main/csv/life_expectancy.csv"
df = pd.read_csv(url).dropna()

X = df[['BMI', 'GDP', 'Alcohol']].values  # 특성
y = df['Life expectancy'].values  # 타겟

# 과대적합을 의도적으로 발생시키기 위해 훈련 데이터를 단 50개만 추출
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
np.random.seed(42)
sample_indices = np.random.choice(len(X_train_full), 50, replace=False)
X_train = X_train_full[sample_indices]
y_train = y_train_full[sample_indices]

# 저장된 모델 및 기준값 불러오기
model_linear = joblib.load('model_linear.pkl')
model_poly = joblib.load('model_poly.pkl')
model_ridge = joblib.load('model_ridge.pkl')

# 성능 평가지표 및 총 특성 개수 테이블
# (1) model_linear
train_pred_linear = model_linear.predict(X_train)
test_pred_linear = model_linear.predict(X_test)

mse_train_linear = mean_squared_error(y_train, train_pred_linear)
mse_test_linear = mean_squared_error(y_test, test_pred_linear)

r2_train_linear = r2_score(y_train, train_pred_linear)
r2_test_linear = r2_score(y_test, test_pred_linear)

# (2) model_poly
train_pred_poly = model_poly.predict(X_train)
test_pred_poly = model_poly.predict(X_test)

mse_train_poly = mean_squared_error(y_train, train_pred_poly)
mse_test_poly = mean_squared_error(y_test, test_pred_poly)

r2_train_poly = r2_score(y_train, train_pred_poly)
r2_test_poly = r2_score(y_test, test_pred_poly)

# (3) model_ridge
train_pred_ridge = model_ridge.predict(X_train)
test_pred_ridge = model_ridge.predict(X_test)

mse_train_ridge = mean_squared_error(y_train, train_pred_ridge)
mse_test_ridge = mean_squared_error(y_test, test_pred_ridge)

r2_train_ridge = r2_score(y_train, train_pred_ridge)
r2_test_ridge = r2_score(y_test, test_pred_ridge)

# DataFrame 생성
metrix = {
    'Model': ['Linear', 'Poly', 'Ridge'],
    'Train R2': [r2_train_linear, r2_train_poly, r2_train_ridge],
    'Test R2': [r2_test_linear, r2_test_poly, r2_test_ridge],
    'Train MSE': [mse_train_linear, mse_train_poly, mse_train_ridge],
    'Test MSE': [mse_test_linear, mse_test_poly, mse_test_ridge],
    'Complexity': [3, 19, 19]  # Linear 3개, Poly와 Ridge 19개
}
df_table = pd.DataFrame(metrix)

# UI for Dashboard
st.set_page_config(page_title="특성별 기대수명 분석", layout="wide")
st.title("모델 별 기대수명 분석 예측")

# 테이블 출력
st.header("모델 별 성능 비교 테이블")
st.dataframe(df_table.set_index('Model'))

# 사이드바 구현
st.sidebar.header("각 특성별 수치 조절")
u_bmi = st.sidebar.slider("BMI", 0, 100, 30)
u_gdp = st.sidebar.slider("GDP", 0, 100000, 10000)
u_alc = st.sidebar.slider("Alcohol", 0, 15, 5)

# 모델 선택 인터페이스
st.sidebar.header("모델 선택")
selected_model = st.sidebar.selectbox("Select Model", ["Linear", "Poly", "Ridge"])

# 실시간 동적 예측 결과 출력
st.header("실시간 기대수명 예측 결과")
model_map = {
    'Linear': model_linear,
    'Poly': model_poly,
    'Ridge': model_ridge
}
target_model = model_map[selected_model]
user_input = np.array([[u_bmi, u_gdp, u_alc]])
prediction = target_model.predict(user_input)
st.write(f"# {prediction[0]:.2f} 세")

# 3종 모델의 Test R^2 점수 비교 막대 그래프
st.header("Test R^2 점수 비교 막대 그래프")
fig, ax = plt.subplots(figsize=(4, 2.5))
bars = ax.bar(df_table['Model'], df_table['Test R2'], width=0.4)
ax.set_xlabel('Model')
ax.set_ylabel('Test R^2')
ax.set_title('Test R^2 Score')
ax.set_ylim((-2, 1))
ax.grid(True, linestyle=':', alpha=0.6)
st.pyplot(fig, use_container_width=False)
