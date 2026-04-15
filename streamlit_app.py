import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# 페이지 설정
st.set_page_config(
    page_title="스마트폰 의존도 분석 대시보드",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 1.5rem;
    }
    .card {
        background: white;
        border-radius: 22px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid #e5e7eb;
    }
    .metric-card {
        background: white;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #e5e7eb;
    }
    .analysis-purpose {
        background: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 8px;
    }
    .interpretation-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        margin-top: 1rem;
        border-radius: 8px;
    }
    .conclusion-card {
        background: #ecfdf5;
        border-left: 4px solid #10b981;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-radius: 12px;
    }
    .sidebar-nav {
        padding: 1rem 0;
    }
    .sidebar-nav a {
        display: block;
        padding: 0.5rem 1rem;
        margin-bottom: 0.5rem;
        text-decoration: none;
        color: #374151;
        border-radius: 8px;
        transition: background-color 0.3s;
    }
    .sidebar-nav a:hover {
        background-color: #e5e7eb;
    }
    .dataframe-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .image-card {
        background: white;
        border-radius: 22px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
        text-align: center;
    }
    [data-testid="stDataFrame"] tbody tr td {
        text-align: center;
    }
    [data-testid="stDataFrame"] thead tr th {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# 사이드바 목차
st.sidebar.title("📋 목차")
st.sidebar.markdown("""
<div class="sidebar-nav">
    <a href="#introduction">서론(연구배경)</a>
    <a href="#variables">변수 설명표</a>
    <a href="#questions">문항표</a>
    <a href="#eda">탐색적 데이터 분석</a>
    <a href="#regression">회귀 결과 해석</a>
    <a href="#conclusion">결론 및 제언</a>
</div>
""", unsafe_allow_html=True)

# 데이터 준비 (하드코딩)
total_data = 2799
missing_data = {
    '스마트폰 의존도 환산점수': {'count': 388, 'percent': 13.86},
    '스마트폰 이용 시간': {'count': 559, 'percent': 19.97},
    '부모의 스마트폰 과다 사용': {'count': 152, 'percent': 5.43},
    '부모의 통제 및 간섭': {'count': 139, 'percent': 4.97},
    '주관적 행복감': {'count': 136, 'percent': 4.86}
}

descriptive_stats = {
    '스마트폰 의존도': {'count': 2411, 'mean': 8.564911, 'std': 3.159034, 'min': 4, '25%': 6, '50%': 8, '75%': 11, 'max': 20},
    '스마트폰 이용 시간': {'count': 2240, 'mean': 3.924107, 'std': 1.531433, 'min': 1, '25%': 3, '50%': 4, '75%': 5, 'max': 6},
    '부모의 스마트폰 과다 사용': {'count': 2647, 'mean': 3.270117, 'std': 1.071493, 'min': 1, '25%': 3, '50%': 3, '75%': 4, 'max': 5},
    '부모의 통제 및 간섭': {'count': 2660, 'mean': 3.368421, 'std': 1.118908, 'min': 1, '25%': 3, '50%': 3, '75%': 4, 'max': 5},
    '주관적 행복감': {'count': 2663, 'mean': 4.114532, 'std': 0.982963, 'min': 1, '25%': 3, '50%': 4, '75%': 5, 'max': 5}
}

correlation_matrix = np.array([
    [1.00, 0.02, -0.15, -0.14],
    [0.02, 1.00, 0.10, -0.07],
    [-0.15, 0.10, 1.00, 0.05],
    [-0.14, -0.07, 0.05, 1.00]
])
variables = ['스마트폰 이용 시간', '부모의 스마트폰 과다 사용', '부모의 통제 및 간섭', '주관적 행복감']

vif_data = {
    'const': 48.641804,
    '스마트폰 이용 시간': 1.043519,
    '부모의 스마트폰 과다 사용': 1.015963,
    '부모의 통제 및 간섭': 1.035792,
    '주관적 행복감': 1.026005
}

regression_results = {
    'model_info': {
        'Dependent Variable': '스마트폰 의존도',
        'No. Observations': 2133,
        'R-squared': 0.128,
        'Adjusted R-squared': 0.126,
        'F-statistic': 78.06,
        'Prob(F-statistic)': 7.44e-62
    },
    'coefficients': {
        'const': {'coef': 7.2641, 'p': 0.000},
        '스마트폰 이용 시간': {'coef': 0.5820, 'p': 0.000},
        '부모의 스마트폰 과다 사용': {'coef': 0.2917, 'p': 0.000},
        '부모의 통제 및 간섭': {'coef': 0.0538, 'p': 0.357},
        '주관적 행복감': {'coef': -0.4918, 'p': 0.000}
    },
    'ci': {
        '스마트폰 이용 시간': [0.499, 0.665],
        '부모의 스마트폰 과다 사용': [0.173, 0.410],
        '부모의 통제 및 간섭': [-0.061, 0.168],
        '주관적 행복감': [-0.620, -0.364]
    }
}

# 샘플 데이터 생성 (시각화용)
np.random.seed(42)
sample_data = pd.DataFrame({
    '스마트폰 이용 시간': np.random.choice([1,2,3,4,5,6], 2133, p=[0.1, 0.15, 0.2, 0.25, 0.2, 0.1]),
    '부모의 스마트폰 과다 사용': np.random.choice([1,2,3,4,5], 2133, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
    '부모의 통제 및 간섭': np.random.choice([1,2,3,4,5], 2133, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
    '주관적 행복감': np.random.choice([1,2,3,4,5], 2133, p=[0.05, 0.1, 0.2, 0.3, 0.35])
})

# Helper 함수들
BASE_DIR = Path(__file__).parent
IMAGE_DIR = BASE_DIR / "images"

def load_image(image_name):
    image_path = IMAGE_DIR / image_name
    if image_path.exists():
        return image_path
    fallback_path = BASE_DIR / image_name
    if fallback_path.exists():
        return fallback_path
    st.warning(f"{image_name} 파일을 찾을 수 없습니다. images 폴더에 위치시켜 주세요.")
    return None

def display_image_card(image_name, title="", caption="", width=650):
    image_path = load_image(image_name)
    if image_path:
        cols = st.columns([1, 3, 1])
        with cols[1]:
            st.image(str(image_path), caption=caption, width=width)

def display_image_grid(images, titles, captions=None):
    if captions is None:
        captions = [""] * len(images)
    cols = st.columns(2)
    for i, (img, title, cap) in enumerate(zip(images, titles, captions)):
        with cols[i % 2]:
            display_image_card(img, title, cap)

# 함수 정의
def introduction():
    st.markdown('<div id="introduction"></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-header">스마트폰 의존도에 영향을 미치는 요소는 무엇인가?</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #6b7280; margin-bottom: 3rem;">초등학생 미디어 이용 실태 자료를 활용한 탐색적 데이터 분석 및 다중선형회귀</h2>', unsafe_allow_html=True)
    
    # 핵심 요약 카드
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("표본 수", "2,799명")
    with col2:
        st.metric("분석 표본", "2,133명")
    with col3:
        st.metric("설명력 (R²)", "12.8%")
    with col4:
        st.metric("유의 변수", "3개")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="card">
            <h4>📱 스마트폰 사용의 일상화</h4>
            <p>현대 사회에서 스마트폰은 필수적인 도구로 자리 잡았으며, 특히 아동·청소년의 일상생활에 깊숙이 침투해 있습니다.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <h4>🏠 가정환경의 영향</h4>
            <p>부모의 미디어 사용 습관과 통제 방식이 자녀의 스마트폰 의존도에 미치는 영향을 탐구합니다.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card">
            <h4>😊 정서적 요인</h4>
            <p>주관적 행복감과 같은 심리적 요인이 스마트폰 사용 패턴에 어떻게 작용하는지 분석합니다.</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="card">
            <h4>📊 데이터 기반 탐색</h4>
            <p>실제 설문조사 데이터를 활용하여 통계적 방법으로 요인 간 관계를 과학적으로 검증합니다.</p>
        </div>
        """, unsafe_allow_html=True)

def variables_table():
    st.markdown('<div id="variables"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">변수 설명표</h2>', unsafe_allow_html=True)
    
    # 종속변수
    st.markdown("""
    <div class="card" style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border-left: 6px solid #3b82f6;">
        <h3 style="color: #1e40af;">종속변수: 스마트폰 의존도</h3>
        <p><strong>문항코드:</strong> 문8의 1~4번 항목 합산 점수</p>
        <p><strong>측정내용:</strong> 스마트폰에 대한 심리적·행동적 의존 정도</p>
        <p><strong>역할:</strong> 분석의 대상 변수로, 독립변수들의 영향력을 측정</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card" style="background: linear-gradient(135deg, #d9f7be 0%, #b7eb8f 100%); border-left: 6px solid #22c55e;">
        <h3 style="color: #166534;">독립변수: 주요 예측 요인</h3>
        <p><strong>대상 변수:</strong> 스마트폰 이용 시간, 부모의 스마트폰 과다 사용, 부모의 통제 및 간섭, 주관적 행복감</p>
        <p><strong>측정목적:</strong> 스마트폰 의존도에 미치는 주요 개인·가정·정서 요인을 검증</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 독립변수들
    variables_df = pd.DataFrame({
        '변수명': ['스마트폰 이용 시간', '부모의 스마트폰 과다 사용', '부모의 통제 및 간섭', '주관적 행복감'],
        '문항코드': ['문4 (C_Q4)', '배문5-5 (C_BQ5N5)', '배문5-3 (C_BQ5N3)', '배문7-10 (C_BQ7N10)'],
        '측정내용': ['일일 스마트폰 사용 시간', '부모의 과도한 스마트폰 사용 정도', '부모의 자녀 통제 및 간섭 수준', '개인의 주관적 행복감 정도'],
        '역할': ['독립변수', '독립변수', '독립변수', '독립변수']
    })
    
    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    st.dataframe(variables_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def questions_table():
    st.markdown('<div id="questions"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">문항표</h2>', unsafe_allow_html=True)
    
    questions_df = pd.DataFrame({
        '문항번호': ['문4', '문8 (1~4)', '배문5-5', '배문5-3', '배문7-10'],
        '변수명': ['스마트폰 이용 시간', '스마트폰 의존도', '부모의 스마트폰 과다 사용', '부모의 통제 및 간섭', '주관적 행복감'],
        '핵심 문항 내용': [
            '하루에 스마트폰을 얼마나 사용하시나요?',
            '스마트폰 없이는 불안하다 / 스마트폰을 자주 확인한다 / 스마트폰 사용을 줄이려 해도 실패한다 / 스마트폰 때문에 일상생활이 방해된다',
            '우리 부모님은 스마트폰을 너무 많이 사용하신다',
            '우리 부모님은 내 스마트폰 사용을 엄격하게 통제하고 간섭하신다',
            '나는 내 삶이 행복하다고 느낀다'
        ],
        '척도 방식': ['1~6 범주형 (시간대)', '1~5 Likert 척도 (합산)', '1~5 Likert 척도', '1~5 Likert 척도', '1~5 Likert 척도']
    })
    
    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    st.dataframe(questions_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("원문 문항 더 보기"):
        st.markdown("""
        - **문4**: 1) 30분 미만, 2) 30분 ~ 1시간, 3) 1 ~ 2시간, 4) 2 ~ 3시간, 5) 3 ~ 4시간, 6) 4시간 이상
        - **문8**: 4개 문항 각각 1~5점 척도로 측정, 합산하여 의존도 점수 산출
        - **배문5-5**: 부모의 스마트폰 과다 사용에 대한 자녀의 인식 (1=전혀 그렇지 않다 ~ 5=매우 그렇다)
        - **배문5-3**: 부모의 통제 및 간섭 정도에 대한 자녀의 인식 (1=전혀 그렇지 않다 ~ 5=매우 그렇다)
        - **배문7-10**: 주관적 행복감 측정 (1=전혀 그렇지 않다 ~ 5=매우 그렇다)
        """)
st.markdown("<br>", unsafe_allow_html=True)

def eda_section():
    st.markdown('<div id="eda"></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">탐색적 데이터 분석 과정</h2>', unsafe_allow_html=True)
    
    # ① 전체 데이터 개수
    st.markdown('<h3> ① 전체 데이터 개수</h3>', unsafe_allow_html=True)
    st.markdown('<div class="analysis-purpose">전체 표본 수를 확인하여 분석의 기본 규모를 파악한다. 이후 결측치 제거 전후의 변화를 비교하기 위한 기준점이다.</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("전체 데이터 개수", f"{total_data:,}")
    st.markdown('<div class="interpretation-box">총 2,799명의 초등학생 응답 데이터로 분석을 진행합니다.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    # ② 결측치 확인
    st.markdown('<h3> ② 결측치 확인</h3>', unsafe_allow_html=True)
    st.markdown('<div class="analysis-purpose">결측치 확인은 분석에 사용할 수 있는 실제 표본 수를 파악하고, 특정 변수에서 응답 누락이 얼마나 큰지 확인하기 위해 필요하다. 결측치가 많으면 분석 결과의 안정성과 해석 가능성에 영향을 줄 수 있다.</div>', unsafe_allow_html=True)
    
    missing_df = pd.DataFrame(missing_data).T.reset_index()
    missing_df.columns = ['변수', '결측치 개수', '결측치 비율(%)']
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
        st.dataframe(missing_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        fig = px.bar(missing_df, x='결측치 비율(%)', y='변수', orientation='h', 
                     title='변수별 결측치 비율', color='결측치 비율(%)', 
                     color_continuous_scale='Blues')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # 4-3. 기초통계
    st.markdown('<h3> ③ 변수에 대한 기초통계 확인</h3>', unsafe_allow_html=True)
    st.markdown('<div class="analysis-purpose">기초통계는 변수의 전반적인 분포 수준과 응답 경향을 파악하기 위해 확인한다. 평균, 중앙값, 표준편차, 최소값과 최대값 등을 통해 데이터가 어느 값대에 몰려 있는지, 얼마나 퍼져 있는지, 그리고 해석에 주의할 만한 특징이 있는지를 미리 살펴볼 수 있다.</div>', unsafe_allow_html=True)
    
    stats_df = pd.DataFrame(descriptive_stats).T.round(3)
    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    st.dataframe(stats_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    # 4-4. 박스플롯 (이미지 사용)
    st.markdown('<h3> ④ 독립변수 세로상자그림</h3>', unsafe_allow_html=True)
    st.markdown('<div class="analysis-purpose">세로상자그림은 변수의 분포, 중앙값, 사분위 범위, 이상치를 한눈에 확인하기 위해 사용한다. 회귀분석 전 변수의 분포 특성을 빠르게 점검하는 데 유용하다.</div>', unsafe_allow_html=True)
    
    boxplot_images = [
        '스마트폰 이용 시간 세로상자그림.png',
        '부모의 스마트폰 과다 사용 세로상자그림.png',
        '부모의 통제 및 간섭 세로상자그림.png',
        '주관적 행복감 세로상자그림.png'
    ]
    boxplot_titles = variables

    boxplot_captions = [
        "스마트폰 이용 시간은 1~6 범주형 변수로, 중앙값 4에 분포",
        "부모의 스마트폰 과다 사용은 3점대에 중앙값이 위치",
        "부모의 통제 및 간섭도 3점대에 많이 분포",
        "주관적 행복감은 4점대에 중앙값이 위치하며, 상대적으로 높은 값에 분포"
    ]
    display_image_grid(boxplot_images, boxplot_titles, boxplot_captions)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # 4-5. 히스토그램 (이미지 사용)
    st.markdown('<h3> ⑤ 독립변수 히스토그램</h3>', unsafe_allow_html=True)
    st.markdown('<div class="analysis-purpose">히스토그램은 변수의 빈도 분포를 확인하여, 응답이 특정 값에 몰려 있는지 살펴보기 위해 필요하다.</div>', unsafe_allow_html=True)
    
    hist_images = [
        '스마트폰 이용시간 히스토그램.png',
        '부모의 스마트폰 과다 사용 히스토그램.png',
        '부모의 통제 및 간섭 히스토그램.png',
        '주관적 행복감 히스토그램.png'
    ]
    hist_titles = variables
    hist_captions = [
        "스마트폰 이용 시간은 4~5점대에 상대적으로 많이 분포",
        "부모의 스마트폰 과다 사용은 3점대에 많이 응답",
        "부모의 통제 및 간섭도 3점대에 집중",
        "주관적 행복감은 4~5점대에 많이 분포"
    ]
    display_image_grid(hist_images, hist_titles, hist_captions)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    # 4-6. 결측치 제거 후 데이터 개수
    st.markdown('<h3> ⑥ 결측치 제거 후 남는 데이터 개수</h3>', unsafe_allow_html=True)
    st.markdown('<div class="analysis-purpose">다중회귀분석에서는 모든 변수값이 갖추어진 사례만 분석에 포함되므로, 결측치를 제거하여 실제 분석 표본을 확정하고 결과의 신뢰성을 높인다.</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("원래 데이터 개수", f"{total_data:,}")
    with col2:
        st.metric("결측치 제거 후 남은 데이터 개수", "2,133")
    with col3:
        st.metric("제거된 데이터 개수", "666")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    # 4-7. 다중공선성 확인
    st.markdown('<h3> ⑦ 다중공선성 확인</h3>', unsafe_allow_html=True)
    st.markdown('<div class="analysis-purpose">다중공선성은 독립변수들끼리 지나치게 높은 상관을 가질 때 발생하며, 회귀계수 해석을 불안정하게 만들 수 있다. 따라서 회귀분석 전에 독립변수 간 상관 수준을 점검할 필요가 있다.</div>', unsafe_allow_html=True)
    
    display_image_card('독립변수간 상관관계 히트맵.png', '독립변수 간 상관행렬 히트맵')
    
    corr_df = pd.DataFrame(correlation_matrix, index=variables, columns=variables)
    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    st.dataframe(corr_df.round(2), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="interpretation-box">전반적으로 상관계수 절댓값이 크지 않아 심각한 다중공선성 징후는 보이지 않는다.</div>', unsafe_allow_html=True)
    
    # 4-8. VIFst.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<h3>⑧ 변수들의 VIF</h3>', unsafe_allow_html=True)
    st.markdown('<div class="analysis-purpose">VIF는 한 독립변수가 다른 독립변수들과 얼마나 중복되는지를 수치로 보여준다. 일반적으로 VIF가 5 미만이면 다중공선성 문제가 크지 않다고 본다.</div>', unsafe_allow_html=True)
    
    vif_df = pd.DataFrame(list(vif_data.items()), columns=['변수', 'VIF'])
    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    st.dataframe(vif_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    fig = px.bar(vif_df[vif_df['변수'] != 'const'], x='변수', y='VIF', title='VIF 값 비교')
    fig.add_hline(y=5, line_dash="dash", line_color="red", annotation_text="VIF=5 기준선")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="interpretation-box">모든 독립변수의 VIF가 약 1 수준으로 매우 낮아, 본 회귀모형에서 다중공선성 문제는 거의 없는 것으로 판단된다.</div>', unsafe_allow_html=True)

def regression_section():
    st.markdown('<div id="regression"></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">다중선형회귀 결과 및 해석</h2>', unsafe_allow_html=True)
    
    # 4-9. 회귀 결과
    st.markdown('<h3> ⑨ 다중선형회귀 결과</h3>', unsafe_allow_html=True)
    
    # 모델 정보
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("종속변수", regression_results['model_info']['Dependent Variable'])
    with col2:
        st.metric("관측치 수", f"{regression_results['model_info']['No. Observations']:,}")
    with col3:
        st.metric("R-squared", f"{regression_results['model_info']['R-squared']:.3f}")
    with col4:
        st.metric("Adj. R-squared", f"{regression_results['model_info']['Adjusted R-squared']:.3f}")
    with col5:
        st.metric("F-statistic", f"{regression_results['model_info']['F-statistic']:.2f}")
    with col6:
        st.metric("Prob(F)", f"{regression_results['model_info']['Prob(F-statistic)']:.2e}")
    
    # 계수 표
    coef_df = pd.DataFrame([
        {'변수': k, '계수': v['coef'], 'p값': v['p'], '유의성': '유의' if v['p'] < 0.05 else '비유의'}
        for k, v in regression_results['coefficients'].items()
    ])
    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    st.dataframe(coef_df.style.apply(lambda x: ['background-color: #d1fae5' if x['유의성'] == '유의' else 'background-color: #fee2e2' for _ in x], axis=1), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 계수 플롯
    coef_plot_df = coef_df[coef_df['변수'] != 'const']
    fig = px.bar(coef_plot_df, x='계수', y='변수', orientation='h', 
                 color='유의성', color_discrete_map={'유의': '#10b981', '비유의': '#ef4444'},
                 title='회귀계수 시각화')
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    # 4-10. 회귀 결과 해석
    st.markdown('<h3> ⑩ 회귀 결과 해석</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h4>모형 설명력 및 유의성</h4>
        <p><strong>R-squared = 0.128:</strong> 이 모델은 스마트폰 의존도의 차이를 약 12.8% 설명한다. 설명력이 아주 높은 수준은 아니지만, 주요 요인들이 일정 부분 관련됨을 보여준다.</p>
        <p><strong>Prob(F-statistic) = 7.44e-62:</strong> p값이 0.05보다 훨씬 작으므로 모형 전체는 통계적으로 유의하다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 변수별 해석 카드
    for var in ['스마트폰 이용 시간', '부모의 스마트폰 과다 사용', '부모의 통제 및 간섭', '주관적 행복감']:
        coef = regression_results['coefficients'][var]['coef']
        p_val = regression_results['coefficients'][var]['p']
        significance = "유의" if p_val < 0.05 else "비유의"
        
        if var == '스마트폰 이용 시간':
            interpretation = f"스마트폰 이용 시간이 1단계 증가할수록 스마트폰 의존도는 평균 약 {coef:.4f}점 증가"
        elif var == '부모의 스마트폰 과다 사용':
            interpretation = f"부모의 스마트폰 과다 사용이 1단계 높아질수록 스마트폰 의존도는 평균 약 {coef:.4f}점 증가"
        elif var == '부모의 통제 및 간섭':
            interpretation = "뚜렷한 영향 요인으로 보기 어려움"
        elif var == '주관적 행복감':
            interpretation = f"주관적 행복감이 1단계 높아질수록 스마트폰 의존도는 평균 약 {abs(coef):.4f}점 감소"
        
        color = "#10b981" if significance == "유의" else "#ef4444"
        st.markdown(f"""
        <div class="card" style="border-left: 6px solid {color};">
            <h4>{var} <span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em;">{significance}</span></h4>
            <p><strong>계수:</strong> {coef:.4f} (p = {p_val:.3f})</p>
            <p>{interpretation}</p>
        </div>
        """, unsafe_allow_html=True)

def conclusion_section():
    st.markdown('<div id="conclusion"></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">결론 및 제언</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="conclusion-card">
        <h4>📊 분석 결과 요약</h4>
        <p>스마트폰 이용 시간, 부모의 스마트폰 과다 사용, 주관적 행복감은 스마트폰 의존도에 유의한 영향을 미치는 변수로 확인되었습니다. 반면, 부모의 통제 및 간섭은 유의한 영향을 미치지 않았습니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card">
            <h4>⏰ 사용 시간이 길수록 의존도 증가</h4>
            <p>스마트폰 이용 시간이 길어질수록 의존도가 높아지는 경향을 보였습니다.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <h4>👨‍👩‍👧‍👦 부모의 과다 사용이 자녀 의존도와 관련</h4>
            <p>부모의 스마트폰 과다 사용이 자녀의 의존도 증가와 연관되어 있습니다.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card">
            <h4>😊 행복감이 높을수록 의존도 감소</h4>
            <p>주관적 행복감이 높은 경우 스마트폰 의존도가 낮아지는 경향을 보였습니다.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="conclusion-card">
        <h4>💡 정책 및 실천 제언</h4>
        <ul>
            <li>청소년 개인의 사용 습관뿐 아니라 가정 내 미디어 환경도 함께 고려할 필요가 있습니다.</li>
            <li>부모의 행동 모델링이 중요할 수 있습니다.</li>
            <li>행복감과 같은 정서적 요인을 함께 살피는 접근이 필요합니다.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("참고 시각화: 스마트폰 의존도 분포"):
        display_image_card('스마트폰 의존도 히스토그램.png', '스마트폰 의존도 히스토그램')
        display_image_card('스마트폰 의존도 세로상자그림.png', '스마트폰 의존도 세로상자그림')

# 메인 앱 실행
def main():
    introduction()
    variables_table()
    questions_table()
    eda_section()
    regression_section()
    conclusion_section()

if __name__ == "__main__":
    main()