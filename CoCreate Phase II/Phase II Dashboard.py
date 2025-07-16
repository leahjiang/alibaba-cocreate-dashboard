import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io

# 页面设置
st.set_page_config(page_title="COCREATE Pitch Phase II 数据看板", layout="wide")

# 数据读取，使用 st.cache_data 缓存以提高性能
@st.cache_data
def load_data():
    """Load the CSV file from the specified path."""
    try:
        df = pd.read_csv("CoCreate Phase II/Update-PitchData-Phase2.csv")
        return df
    except FileNotFoundError:
        st.error("无法找到 'Update-PitchData-Phase2.csv' 文件。请确保文件与脚本在同一目录下。")
        return pd.DataFrame()

df = load_data()

# 检查数据是否成功加载
if not df.empty:
    st.title("📊 COCREATE Pitch Phase II 数据看板")
    st.markdown("基于2,685条报名数据的核心指标与可视化分析")

    # ----------------------------
    # 1. 项目数据总览
    # ----------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📌 总报名数", len(df))
    col2.metric("✅ 完整提交数", df[df['Response Type'] == 'complete'].shape[0])
    col3.metric("📈 完成率", f"{df[df['Response Type'] == 'complete'].shape[0] / len(df) * 100:.1f}%")
    col4.metric("🌍 国家数量", df['country'].nunique())

    st.markdown("---")

    # ----------------------------
    # 2. 渠道来源分析
    # ----------------------------
    st.header("📣 渠道来源分析")

    col1, col2 = st.columns([1, 1.2])
    with col1:
        channel_counts = df['渠道'].value_counts().reset_index()
        channel_counts.columns = ['渠道', '数量']
        fig_channel = px.pie(channel_counts, names='渠道', values='数量', title="整体渠道分布")
        st.plotly_chart(fig_channel, use_container_width=True)

    with col2:
        selected_channel = st.selectbox("选择一个渠道以查看其下 SOURCE 分布：", df['渠道'].dropna().unique(), key='channel_select')
        filtered_df_source = df[df['渠道'] == selected_channel]
        source_counts = filtered_df_source['SOURCE'].value_counts().reset_index()
        source_counts.columns = ['SOURCE', '数量']
        fig_source = px.pie(source_counts, names='SOURCE', values='数量', title=f"{selected_channel} 渠道下的 SOURCE 分布")
        st.plotly_chart(fig_source, use_container_width=True)

    st.markdown("---")

    # ----------------------------
    # 3. 地理分布分析
    # ----------------------------
    st.header("🌍 国家分布分析")
    country_counts = df['country'].value_counts().reset_index()
    country_counts.columns = ['国家', '数量']
    fig_country = px.pie(country_counts.head(10), names='国家', values='数量', title="参赛公司 Top 10 国家分布")
    st.plotly_chart(fig_country, use_container_width=True)
    st.markdown("---")

    # ----------------------------
    # 4. 行业与阶段分析
    # ----------------------------
    st.header("🏢 行业与发展阶段分析")
    col1, col2 = st.columns(2)
    with col1:
        industry_col = 'Which of the following industries best describes your company?'
        if industry_col in df.columns:
            industry_counts = df[industry_col].value_counts().reset_index()
            industry_counts.columns = ['行业', '数量']
            fig_industry = px.bar(industry_counts.head(10), x='数量', y='行业', orientation='h', title="Top 10 行业分布")
            fig_industry.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_industry, use_container_width=True)
        else:
            st.warning(f"缺少字段：'{industry_col}'")

    with col2:
        stage_col = 'What stage is your company currently in?'
        if stage_col in df.columns:
            stage_counts = df[stage_col].value_counts().reset_index()
            stage_counts.columns = ['发展阶段', '数量']
            fig_stage = px.bar(stage_counts, x='发展阶段', y='数量', title="公司发展阶段分布")
            st.plotly_chart(fig_stage, use_container_width=True)
        else:
            st.warning(f"缺少字段：'{stage_col}'")
    st.markdown("---")

    # ----------------------------
    # 5. 公司类型与产品分析
    # ----------------------------
    st.header("💼 公司类型与产品类型分析")
    col1, col2 = st.columns(2)

    with col1:
        company_type_col = 'My company is a'
        if company_type_col in df.columns:
            company_type_counts = df[company_type_col].value_counts().reset_index()
            company_type_counts.columns = ['公司类型', '数量']
            fig_company_type = px.pie(company_type_counts, names='公司类型', values='数量', title="公司类型分布", hole=0.3)
            st.plotly_chart(fig_company_type, use_container_width=True)
        else:
            st.warning(f"缺少字段：'{company_type_col}'")
            
    with col2:
        st.subheader("产品类型统计")
        product_types = [
            'Physical Products - Tangible goods that can be sold/distributed online',
            'Digital Products - Software, apps, or digital solutions',
            'Hardware + Software - Physical devices with digital components',
            'Digital Services - Online platforms, marketplaces, or service delivery',
            'Professional Services - Consulting, advisory, or traditional services'
        ]
        
        product_data = {}
        for p_type_col in product_types:
            if p_type_col in df.columns:
                product_data[p_type_col.split(' - ')[0]] = df[p_type_col].isin(['Yes', True]).sum()

        if product_data:
            product_df = pd.DataFrame(list(product_data.items()), columns=['产品类型', '数量'])
            fig_product_type = px.bar(product_df, x='数量', y='产品类型', orientation='h', title="产品类型统计")
            fig_product_type.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_product_type, use_container_width=True)
        else:
            st.warning("缺少产品类型相关字段。")

    st.markdown("---")
    
    # ----------------------------
    # 6. 平台账号与用户反馈
    # ----------------------------
    st.header("🤝 平台账号与用户反馈")
    alibaba_account_col = 'Do you have an Alibaba.com account?'
    if alibaba_account_col in df.columns:
        alibaba_account_counts = df[alibaba_account_col].value_counts().reset_index()
        alibaba_account_counts.columns = ['是否有 Alibaba.com 账号', '数量']
        fig_alibaba = px.pie(alibaba_account_counts, names='是否有 Alibaba.com 账号', values='数量', title="是否有 Alibaba.com 账号", hole=0.3)
        st.plotly_chart(fig_alibaba, use_container_width=True)
    else:
        st.warning(f"缺少字段：'{alibaba_account_col}'")
    st.markdown("---")

    # ----------------------------
    # 7. 创业故事与产品方案分析 (关键词云)
    # ----------------------------
    st.header("💡 创业故事与产品方案分析 (关键词云)")
    
    text_fields = {
        "Describe your solution and explain your key competitive advantages compared to existing alternatives": "产品方案",
        "Describe your business story that you’d like to share with us": "商业故事",
        "What specific market problem does your company aim to solve?": "解决的问题"
    }

    selected_text_field_name = st.selectbox("选择文本字段以生成关键词云：", list(text_fields.keys()), key='text_field_select')
    selected_label = text_fields.get(selected_text_field_name, "关键词")
    
    text_content = df[selected_text_field_name].dropna().astype(str).tolist()
    if text_content:
        text_combined = " ".join(text_content)
        words = re.findall(r'\b\w+\b', text_combined.lower())
        # Placeholder for stop words, use a real list for a production app
        stop_words = set(px.data.stopwords.words()) 
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        if filtered_words:
            word_freq = Counter(filtered_words)
            wc = WordCloud(width=800, height=400, background_color="white", collocations=False).generate_from_frequencies(word_freq)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            ax.set_title(f"{selected_label} 关键词云")
            st.pyplot(fig)
        else:
            st.info(f"选定字段 '{selected_label}' 没有足够的文本内容生成关键词云。")
    else:
        st.info(f"选定字段 '{selected_label}' 没有可用的文本内容。")

    st.markdown("---")

    # ----------------------------
    # 8. 数据筛选 + 表格导出
    # ----------------------------
    st.header("🔍 数据筛选与导出")
    st.markdown("使用以下筛选器查看特定数据，并可将筛选后的数据导出为 CSV。")

    col_filters = st.columns(4)
    with col_filters[0]:
        if 'country' in df.columns:
            selected_country = st.multiselect("选择国家", df['country'].unique(), key='filter_country')
    with col_filters[1]:
        if '渠道' in df.columns:
            selected_channel_filter = st.multiselect("选择渠道", df['渠道'].unique(), key='filter_channel')
    with col_filters[2]:
        if 'Response Type' in df.columns:
            selected_response_type = st.multiselect("选择完成状态", df['Response Type'].unique(), key='filter_response_type')
    with col_filters[3]:
        capital_col = 'Has your company secured funding?'
        if capital_col in df.columns:
            selected_capital_raised = st.multiselect("是否融资", df[capital_col].unique(), key='filter_capital_raised')

    filtered_df = df.copy()

    if 'selected_country' in locals() and selected_country:
        filtered_df = filtered_df[filtered_df['country'].isin(selected_country)]
    if 'selected_channel_filter' in locals() and selected_channel_filter:
        filtered_df = filtered_df[filtered_df['渠道'].isin(selected_channel_filter)]
    if 'selected_response_type' in locals() and selected_response_type:
        filtered_df = filtered_df[filtered_df['Response Type'].isin(selected_response_type)]
    if 'selected_capital_raised' in locals() and selected_capital_raised:
        filtered_df = filtered_df[filtered_df[capital_col].isin(selected_capital_raised)]

    st.subheader(f"筛选结果 ({len(filtered_df)} 条记录)")
    st.dataframe(filtered_df)

    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv_data = convert_df_to_csv(filtered_df)
    st.download_button(
        label="📥 导出筛选后的数据为 CSV",
        data=csv_data,
        file_name="filtered_pitch_data.csv",
        mime="text/csv",
    )
    st.markdown("---")
