import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import streamlit as st
import plotly.express as px

st.set_page_config(layout="wide")

st.title("🧠 3D Taste Coding Viewer (Gustatory Cortex Simulation)")
st.markdown("과일의 단맛, 신맛, 쓴맛, 감칠맛 반응값을 기반으로 뇌에서 맛을 어떻게 구분할 수 있는지를 3D로 시각화한 모델입니다.")

# 기본 과일 데이터
fruits = ['살구', '복숭아', '딸기', '사과', '파인애플', '망고', '포도']
data = np.array([
    [10, 0.6, 0.05, 0.02],
    [9, 0.5, 0.04, 0.03],
    [7.5, 1.0, 0.02, 0.01],
    [10, 0.3, 0.03, 0.02],
    [13, 1.2, 0.05, 0.03],
    [14, 0.4, 0.04, 0.04],
    [15, 0.2, 0.03, 0.02],
])
columns = ['단맛', '신맛', '쓴맛', '감칠맛']
df = pd.DataFrame(data, columns=columns)
df['과일'] = fruits

# 사용자 추가 입력
st.sidebar.header("➕ 새로운 과일 추가")
name = st.sidebar.text_input("과일 이름")
sweet = st.sidebar.slider("단맛", 0.0, 20.0, 10.0)
sour = st.sidebar.slider("신맛", 0.0, 2.0, 0.5)
bitter = st.sidebar.slider("쓴맛", 0.0, 0.1, 0.03)
umami = st.sidebar.slider("감칠맛", 0.0, 0.1, 0.03)

if st.sidebar.button("과일 추가") and name:
    new_row = pd.DataFrame([[sweet, sour, bitter, umami, name]], columns=columns + ['과일'])
    df = pd.concat([df, new_row], ignore_index=True)

# 정규화 및 PCA
norm_df = df.copy()
norm_df[columns] = norm_df[columns] / norm_df[columns].max()
pca = PCA(n_components=3)
components = pca.fit_transform(norm_df[columns])

# 시각화 데이터프레임
pca_df = pd.DataFrame(components, columns=['PC1', 'PC2', 'PC3'])
pca_df['과일'] = df['과일']
pca_df['단맛'] = df['단맛']
pca_df['신맛'] = df['신맛']
pca_df['쓴맛'] = df['쓴맛']
pca_df['감칠맛'] = df['감칠맛']

# Plotly 3D 그래프
fig = px.scatter_3d(
    pca_df, x='PC1', y='PC2', z='PC3', color='과일', text='과일',
    hover_data=['단맛', '신맛', '쓴맛', '감칠맛'],
    width=900, height=700
)
fig.update_traces(marker=dict(size=8, opacity=0.8))
fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))

st.plotly_chart(fig, use_container_width=True)

# 데이터 보기
with st.expander("📋 원본 데이터 보기"):
    st.dataframe(df.reset_index(drop=True), use_container_width=True)
