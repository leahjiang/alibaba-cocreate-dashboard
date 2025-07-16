
import streamlit as st
import pandas as pd
import plotly.express as px

# 读取CSV数据
@st.cache_data
def load_data():
    df = pd.read_csv("0714-Pitch Source- Phase1.csv")
    df = df.rename(columns={
        'Country where {{field:4b95525c-36f9-47c2-b2e9-50b3e64a92cb}} is based:': 'Country',
        '渠道分类': 'Channel',
        'SOURCE': 'Source',
        'Response Type': 'ResponseType',
        'Company Name': 'Company'
    })
    return df

# 初始化
st.set_page_config(page_title="COCREATE Pitch Phase I Dashboard", layout="wide")
st.title("📊 COCREATE Pitch Phase I 数据看板")
df = load_data()

# 概览区域
st.subheader("1. 数据总览")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("总提交数", len(df))
with col2:
    st.metric("完成提交", df[df['ResponseType'] == 'completed'].shape[0])
with col3:
    pct = df[df['ResponseType'] == 'completed'].shape[0] / len(df) * 100
    st.metric("完成率", f"{pct:.1f}%")

# 渠道分类图
st.subheader("2. 渠道分类分布")
channel_count = df['Channel'].value_counts().reset_index()
channel_count.columns = ['Channel', 'Count']
fig1 = px.pie(channel_count, names='Channel', values='Count', title='渠道来源占比', hole=0.4)
st.plotly_chart(fig1, use_container_width=True)

# 国家分布图
st.subheader("3. 国家分布 Top 10")
country_count = df['Country'].value_counts().nlargest(10).reset_index()
country_count.columns = ['Country', 'Count']
fig2 = px.bar(country_count, x='Country', y='Count', color='Country', title='国家项目数量 Top 10')
st.plotly_chart(fig2, use_container_width=True)

# 详细列表
st.subheader("4. 项目数据总览（可筛选）")
with st.expander("点击展开查看完整项目数据表"):
    selected_country = st.selectbox("按国家筛选：", options=["全部"] + sorted(df['Country'].dropna().unique().tolist()))
    if selected_country != "全部":
        filtered_df = df[df['Country'] == selected_country]
    else:
        filtered_df = df
    st.dataframe(filtered_df[['First Name', 'Last Name', 'Company', 'Country', 'Channel', 'Source', 'ResponseType']], use_container_width=True)

# 底部
st.markdown("---")
st.markdown("© 2025 Alibaba.com COCREATE Pitch Phase I Dashboard")
