import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import nltk
import ssl  # Import ssl module

# Bypass SSL certificate verification for NLTK downloads if necessary (common issue on some systems)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Ensure NLTK stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Page settings
st.set_page_config(page_title="COCREATE Pitch Phase II 数据看板", layout="wide")

# Data loading and preprocessing
@st.cache_data
def load_data():
    """Load the CSV file and perform necessary data cleaning/normalization."""
    # UPDATED FILE PATH AND NAME
    file_path = "CoCreate Phase II/Update-PitchData-Phase2.csv"
    try:
        df = pd.read_csv(file_path)

        # --- Data Cleaning and Normalization ---

        # 2. Corrected "Complete" count: "partial" is partial, "complete" or "completed" are complete.
        if 'Response Type' in df.columns:
            df['Response Type Cleaned'] = df['Response Type'].apply(
                lambda x: 'complete' if pd.notna(x) and str(x).lower() in ['complete', 'completed'] else 'partial'
            )
        else:
            df['Response Type Cleaned'] = 'unknown'  # Fallback if column is missing or handled as default partial

        # 7. "是否有 Alibaba.com 账号" - Normalization
        alibaba_col = 'Do you have an Alibaba.com account?'
        if alibaba_col in df.columns:
            df['Alibaba Account Status'] = df[alibaba_col].apply(
                lambda x: 'Yes' if pd.notna(x) and str(x).lower() not in ['no', 'n', 'na', 'n/a', 'n/a.', 'not sure', 'none'] else 'No'
            )
        else:
            df['Alibaba Account Status'] = 'Unknown'  # Fallback

        return df
    except FileNotFoundError:
        st.error(f"无法找到 '{file_path}' 文件。请确保文件路径与脚本所在目录结构匹配。")
        return pd.DataFrame()  # Return empty DataFrame on error

df = load_data()

# Check if data was loaded successfully
if not df.empty:
    # 1. Dynamic "2685" count in title
    st.title(f"📊 COCREATE Pitch Phase II 数据看板")
    st.markdown(f"查看 {len(df)} 条报名数据的核心指标与可视化分析")

    # ----------------------------
    # 1. 项目数据总览
    # ----------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📌 总报名数", len(df))
    # Using the cleaned response type column
    col2.metric("✅ 完整提交数", df[df['Response Type Cleaned'] == 'complete'].shape[0])
    col3.metric("📈 完成率", f"{df[df['Response Type Cleaned'] == 'complete'].shape[0] / len(df) * 100:.1f}%")
    col4.metric("🌍 国家数量", df['country'].nunique() if 'country' in df.columns else 0)

    st.markdown("---")

    # ----------------------------
    # 2. 渠道来源分析
    # ----------------------------
    st.header("📣 渠道来源分析")

    col1, col2 = st.columns([1, 1.2])
    with col1:
        channel_counts = df['渠道'].value_counts().reset_index()
        channel_counts.columns = ['渠道', '数量']
        fig_channel = go.Figure(data=[go.Pie(
            labels=channel_counts['渠道'],
            values=channel_counts['数量'],
            textinfo='label+percent',
            insidetextorientation='radial',
            hole=0.3
        )])
        fig_channel.update_layout(title_text="整体渠道分布")
        st.plotly_chart(fig_channel, use_container_width=True)
        
    with col2:
        selected_channel = st.selectbox("选择一个渠道以查看其下 SOURCE 分布：", df['渠道'].dropna().unique(), key='channel_select')
        filtered_df_source = df[df['渠道'] == selected_channel]
        source_counts = filtered_df_source['SOURCE'].value_counts().reset_index()
        source_counts.columns = ['SOURCE', '数量']
        if source_counts['数量'].sum() == 0:
            st.info(f"{selected_channel} 渠道下无有效的 SOURCE 数据。")
        else:
            fig_source = go.Figure(data=[go.Pie(
                labels=source_counts['SOURCE'],
                values=source_counts['数量'],
                textinfo='label+percent',
                insidetextorientation='radial',
                hole=0.3
            )])
            fig_source.update_layout(title_text=f"{selected_channel} 渠道下的 SOURCE 分布")
            st.plotly_chart(fig_source, use_container_width=True)
            
    st.markdown("---")

    # ----------------------------
    # 3. 地理分布分析
    # ----------------------------
    st.header("🌍 国家分布分析")

    if 'country' in df.columns:
        # 4. "参赛公司 Top 10 国家分布" 柱状图
        country_counts = df['country'].dropna().value_counts().reset_index()  # Dropna
        country_counts.columns = ['国家', '数量']
        if not country_counts.empty and '国家' in country_counts.columns and '数量' in country_counts.columns:
            fig_country_bar = px.bar(country_counts.head(10), x='国家', y='数量', title="参赛公司 Top 10 国家分布",
                                      color='数量', color_continuous_scale=px.colors.sequential.Viridis)
            fig_country_bar.update_layout(xaxis_title=None, yaxis_title=None)
            # 添加参考线：计算前10国家数量的平均值
            avg_value = country_counts.head(10)['数量'].mean()
            fig_country_bar.add_shape(
                type="line",
                x0=avg_value, x1=avg_value,
                y0=-0.5, y1=9.5,
                line=dict(color="red", dash="dash")
            )
            fig_country_bar.add_annotation(
                x=avg_value,
                y=9.5,
                text=f"平均值: {avg_value:.1f}",
                showarrow=False,
                font=dict(color="red"),
                xanchor="left"
            )
            st.plotly_chart(fig_country_bar, use_container_width=True)
        else:
            st.info("没有可用的国家数据进行分析。")

        st.subheader("重点国家分析：美国、英国、德国、法国、意大利的数量与渠道")
        key_countries = ['United States', 'United Kingdom', 'Germany', 'France', 'Italy']
        
        # Filter for key countries and their channel distribution
        key_country_df = df[df['country'].isin(key_countries)]
        
        # Display counts for key countries with most common channel
        st.write("---")
        st.markdown("##### 重点国家报名数量:")
        key_country_summary = key_country_df.groupby('country').agg(
            报名数量=('country', 'count'),
            最主要渠道=('渠道', lambda x: x.value_counts().idxmax() if not x.empty else '无')
        ).reset_index().rename(columns={'country': '国家'})
        st.dataframe(key_country_summary.set_index('国家'))
        
        st.write("---")
        # Display channel distribution for each key country
        st.markdown("##### 重点国家渠道分布:")
        available_key_countries = [c for c in key_countries if c in key_country_df['country'].unique()]
        if available_key_countries and '渠道' in df.columns:
            cols_key_countries = st.columns(len(available_key_countries))
            for i, country_name in enumerate(available_key_countries):
                with cols_key_countries[i]:
                    country_channel_data = key_country_df[key_country_df['country'] == country_name]['渠道'].dropna().value_counts().reset_index()
                    country_channel_data.columns = ['渠道', '数量']
                    if not country_channel_data.empty:
                        fig_country_channel = go.Figure(data=[go.Pie(
                            labels=country_channel_data['渠道'],
                            values=country_channel_data['数量'],
                            textinfo='percent+label',
                            insidetextorientation='radial',
                            hole=0.3
                        )])
                        fig_country_channel.update_layout(title_text=f"{country_name} 渠道分布")
                        st.plotly_chart(fig_country_channel, use_container_width=True)
                    else:
                        st.info(f"没有 {country_name} 的渠道数据。")
        else:
            st.info("没有足够的数据或 '渠道' 字段缺失，无法显示重点国家渠道分布。")
    else:
        st.warning("缺少字段：'country'，无法显示地理分布分析。")

    st.markdown("---")

    # ----------------------------
    # 4. 行业与发展阶段分析
    # ----------------------------
    st.header("🏢 行业与发展阶段分析")

    col1, col2 = st.columns(2)

    with col1:
        industry_col = 'Which of the following industries best describes your company?'
        if industry_col in df.columns:
            industry_counts = df[industry_col].dropna().value_counts().reset_index()  # Dropna
            industry_counts.columns = ['行业', '数量']
            if not industry_counts.empty:
                fig_industry = px.bar(industry_counts.head(10), x='数量', y='行业', orientation='h', title="Top 10 行业分布")
                fig_industry.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title=None, yaxis_title=None)
                # 添加参考线：计算前10行业数量的平均值
                avg_value_industry = industry_counts.head(10)['数量'].mean()
                fig_industry.add_shape(
                    type="line",
                    x0=avg_value_industry,
                    x1=avg_value_industry,
                    y0=-0.5,
                    y1=9.5,
                    line=dict(color="red", dash="dash")
                )
                fig_industry.add_annotation(
                    x=avg_value_industry,
                    y=9.5,
                    text=f"平均值: {avg_value_industry:.1f}",
                    showarrow=False,
                    font=dict(color="red"),
                    xanchor="left"
                )
                st.plotly_chart(fig_industry, use_container_width=True)
            else:
                st.info(f"缺少字段：'{industry_col}' 的数据。")
        else:
            st.warning(f"缺少字段：'{industry_col}'，无法显示行业分析。")

    with col2:
        stage_col = 'What stage is your company currently in?'
        if stage_col in df.columns:
            stage_counts = df[stage_col].dropna().value_counts().reset_index()  # Dropna
            stage_counts.columns = ['发展阶段', '数量']
            if not stage_counts.empty:
                fig_stage = px.bar(stage_counts, x='发展阶段', y='数量', title="公司发展阶段分布")
                fig_stage.update_layout(xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig_stage, use_container_width=True)
            else:
                st.info(f"缺少字段：'{stage_col}' 的数据。")
        else:
            st.warning(f"缺少字段：'{stage_col}'，无法显示发展阶段分析。")

    st.markdown("---")

    # ----------------------------
    # 5. 公司类型与产品分析
    # ----------------------------
    st.header("💼 公司类型与产品类型分析")

    col1, col2 = st.columns(2)

    with col1:
        company_type_col = 'My company is a:'
        if company_type_col in df.columns:
            company_type_counts = df[company_type_col].dropna().value_counts().reset_index()  # Dropna
            company_type_counts.columns = ['公司类型', '数量']
            if not company_type_counts.empty:
                fig_company_type = px.pie(company_type_counts, names='公司类型', values='数量',
                                          title="公司类型分布", hole=0.3, textinfo='percent+label')
                st.plotly_chart(fig_company_type, use_container_width=True)
            else:
                st.info(f"缺少字段：'{company_type_col}' 的数据。")
        else:
            st.warning(f"缺少字段：'{company_type_col}'，无法显示公司类型分析。")

    with col2:
        st.subheader("产品类型统计")
        product_types_cols = {
            'Physical Product': 'Physical Products - Tangible goods that can be sold/distributed online',
            'Digital Product': 'Digital Products - Software, apps, or digital solutions',
            'Hardware + Software': 'Hardware + Software - Physical devices with digital components',
            'Digital Service': 'Digital Services - Online platforms, marketplaces, or service delivery',
            'Professional Service': 'Professional Services - Consulting, advisory, or traditional services'
        }
        
        product_data = {}
        for display_name, original_col_name in product_types_cols.items():
            if original_col_name in df.columns:
                # Assuming 'Yes' or boolean True indicates selection
                product_data[display_name] = df[original_col_name].dropna().apply(lambda x: str(x).lower() == 'yes' or x == True).sum()
        
        if product_data:
            product_df = pd.DataFrame(list(product_data.items()), columns=['产品类型', '数量'])
            if not product_df.empty:
                fig_product_type = px.bar(product_df, x='数量', y='产品类型', orientation='h', title="产品类型统计")
                fig_product_type.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_product_type, use_container_width=True)
            else:
                st.info("没有可用的产品类型数据。")
        else:
            st.warning("缺少产品类型相关字段，无法显示产品类型分析。")

    st.markdown("---")

    # ----------------------------
    # 6. 平台账号与用户反馈
    # ----------------------------
    st.header("🤝 平台账号与用户反馈")
    
    alibaba_account_col = 'Alibaba Account Status'
    if alibaba_account_col in df.columns:
        alibaba_account_counts = df[alibaba_account_col].dropna().value_counts().reset_index()
        alibaba_account_counts.columns = ['是否有 Alibaba.com 账号', '数量']
        if not alibaba_account_counts.empty:
            fig_alibaba = px.pie(alibaba_account_counts, names='是否有 Alibaba.com 账号', values='数量',
                                 title="是否有 Alibaba.com 账号", hole=0.3, textinfo='percent+label')
            st.plotly_chart(fig_alibaba, use_container_width=True)
        else:
            st.info(f"缺少字段：'{alibaba_account_col}' 的数据。")
    else:
        st.warning(f"缺少字段：'{alibaba_account_col}'，无法显示 Alibaba.com 账号分析。")

    feedback_content_col = 'Do you have any feedback for Alibaba.com?'
    if feedback_content_col in df.columns:
        st.subheader("部分用户反馈内容")
        sample_feedback_df = df[feedback_content_col].dropna().sample(min(5, len(df[feedback_content_col].dropna()))).tolist()
        if sample_feedback_df:
            for i, fb in enumerate(sample_feedback_df):
                st.write(f"- {fb}")
        else:
            st.info("暂无用户反馈内容。")
    else:
        st.warning(f"缺少字段：'{feedback_content_col}'，无法显示用户反馈内容。")

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

    available_text_fields = {k: v for k, v in text_fields.items() if k in df.columns}

    if available_text_fields:
        selected_text_field_name = st.selectbox("选择文本字段以生成关键词云：", list(available_text_fields.keys()), key='text_field_select')
        selected_label = available_text_fields.get(selected_text_field_name, "关键词")
        
        text_content = df[selected_text_field_name].dropna().astype(str).tolist()
        if text_content:
            text_combined = " ".join(text_content)
            stop_words = set(nltk.corpus.stopwords.words('english'))
            words = re.findall(r'\b\w+\b', text_combined.lower())
            filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
            
            if filtered_words:
                word_freq = Counter(filtered_words)
                wc = WordCloud(width=800, height=400, background_color="white", collocations=False).generate_from_frequencies(word_freq)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                ax.set_title(f"{selected_label} 关键词云")
                st.pyplot(fig)
                
                st.subheader(f"'{selected_label}' 文本样本")
                sample_texts = df[selected_text_field_name].dropna().sample(min(3, len(df[selected_text_field_name].dropna()))).tolist()
                for i, text in enumerate(sample_texts):
                    st.write(f"**样本 {i+1}:**")
                    st.write(text)
            else:
                st.info(f"选定字段 '{selected_label}' 没有足够的文本内容生成关键词云。")
        else:
            st.info(f"选定字段 '{selected_label}' 没有可用的文本内容。")
    else:
        st.warning("数据中不包含任何可用于关键词云分析的文本字段。")

    st.markdown("---")

    # ----------------------------
    # 8. 数据筛选 + 表格导出
    # ----------------------------
    st.header("🔍 数据筛选与导出")
    st.markdown("使用以下筛选器查看特定数据，并可将筛选后的数据导出为 CSV。")

    col_filters = st.columns(4)

    selected_country = []
    selected_channel_filter = []
    selected_response_type = []
    selected_capital_raised = []

    with col_filters[0]:
        if 'country' in df.columns:
            selected_country = st.multiselect("选择国家", df['country'].unique(), key='filter_country')
    with col_filters[1]:
        if '渠道' in df.columns:
            selected_channel_filter = st.multiselect("选择渠道", df['渠道'].unique(), key='filter_channel')
    with col_filters[2]:
        if 'Response Type Cleaned' in df.columns:
            selected_response_type = st.multiselect("选择完成状态", df['Response Type Cleaned'].unique(), key='filter_response_type')
    with col_filters[3]:
        capital_col = 'Has your company secured funding?'
        if capital_col in df.columns:
            selected_capital_raised = st.multiselect("是否融资", df[capital_col].unique(), key='filter_capital_raised')
        else:
            st.warning(f"缺少字段：'{capital_col}'，无法提供融资筛选。")

    filtered_df = df.copy()
    if selected_country:
        filtered_df = filtered_df[filtered_df['country'].isin(selected_country)]
    if selected_channel_filter:
        filtered_df = filtered_df[filtered_df['渠道'].isin(selected_channel_filter)]
    if selected_response_type:
        filtered_df = filtered_df[filtered_df['Response Type Cleaned'].isin(selected_response_type)]
    if selected_capital_raised and capital_col in df.columns:
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

else:
    st.info("数据加载失败，请检查文件路径与文件名是否正确，例如：`CoCreate Phase II/Update-PitchData-Phase2.csv`。")
