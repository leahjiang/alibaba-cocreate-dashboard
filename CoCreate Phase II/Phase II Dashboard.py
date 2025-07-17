import streamlit as st
import pandas as pd
import plotly.express as px # Still import for color scales if needed, but charts will use go
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

# Global column renaming for consistent use in the dashboard
if not df.empty:
    df = df.rename(columns={
        'What stage is your company currently in?': '公司发展阶段',
        "What is your company's current annual revenue?": '公司营收',
        'How many employees/contractors are currently working at your company?': '团队规模',
        # Assuming 'My company is a:' is the correct column name without a colon
        'My company is a:': '公司类型'
    })


# Check if data was loaded successfully
if not df.empty:
    # 1. Dynamic "2685" count in title
    st.title(f"📊 COCREATE Pitch Phase II 数据看板")
    st.markdown(f"查看 {len(df)} 条报名数据的核心指标与可视化分析")

    # ----------------------------
    # 1. 项目数据总览
    # ----------------------------
    st.header("📊 项目数据总览")
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

    if '渠道' in df.columns:
        with col1:
            # Drop NA values before value_counts to ensure only valid channels are counted
            channel_counts = df['渠道'].dropna().value_counts().reset_index()
            channel_counts.columns = ['渠道', '数量']

            # --- DEBUGGING INFORMATION (Keep for now if issues persist) ---
            # st.write("Debug: channel_counts DataFrame:")
            # st.dataframe(channel_counts)
            # st.write("Debug: channel_counts dtypes:")
            # st.write(channel_counts.dtypes)
            # --- END DEBUGGING INFORMATION ---

            # Check if channel_counts has data and the required columns for plotting
            if not channel_counts.empty and '渠道' in channel_counts.columns and '数量' in channel_counts.columns:
                fig_channel = go.Figure(data=[go.Pie(
                    labels=channel_counts['渠道'],
                    values=channel_counts['数量'],
                    textinfo='label+percent',
                    insidetextorientation='radial',
                    hole=0.3
                )])
                fig_channel.update_layout(title_text="整体渠道分布")
                st.plotly_chart(fig_channel, use_container_width=True)
            else:
                st.info("没有可用的渠道数据进行分析。请检查 CSV 文件中的 '渠道' 列是否有数据。")

        with col2:
            unique_channels = df['渠道'].dropna().unique()
            if unique_channels.size > 0:
                selected_channel = st.selectbox("选择一个渠道以查看其下 SOURCE 分布：", unique_channels, key='channel_select')
                filtered_df_source = df[df['渠道'] == selected_channel]
                if 'SOURCE' in df.columns:
                    source_counts = filtered_df_source['SOURCE'].dropna().value_counts().reset_index()
                    source_counts.columns = ['SOURCE', '数量']
                    if not source_counts.empty and 'SOURCE' in source_counts.columns and '数量' in source_counts.columns:
                        fig_source = go.Figure(data=[go.Pie(
                            labels=source_counts['SOURCE'],
                            values=source_counts['数量'],
                            textinfo='label+percent',
                            insidetextorientation='radial',
                            hole=0.3
                        )])
                        fig_source.update_layout(title_text=f"{selected_channel} 渠道下的 SOURCE 分布")
                        st.plotly_chart(fig_source, use_container_width=True)
                    else:
                        st.info(f"没有 {selected_channel} 渠道下的 SOURCE 数据。")
                else:
                    st.warning(f"缺少字段：'SOURCE'，无法显示 {selected_channel} 渠道下的 SOURCE 分布。")
            else:
                st.info("没有可用的渠道供选择。")
    else:
        st.warning("缺少字段：'渠道'，无法显示渠道来源分析。")

    st.markdown("---")

    # ----------------------------
    # 3. 地理分布分析
    # ----------------------------
    st.header("🌍 国家分布分析")

    if 'country' in df.columns:
        # 4. "参赛公司 Top 10 国家分布" 柱状图
        country_counts = df['country'].dropna().value_counts().reset_index()
        country_counts.columns = ['国家', '数量']
        if not country_counts.empty and '国家' in country_counts.columns and '数量' in country_counts.columns:
            fig_country_bar = go.Figure(data=[go.Bar(
                x=country_counts.head(10)['国家'],
                y=country_counts.head(10)['数量'],
                marker_color=country_counts.head(10)['数量'], # Color by count
                marker_colorscale=px.colors.sequential.Viridis # Use px's color scale
            )])
            fig_country_bar.update_layout(title_text="参赛公司 Top 10 国家分布", xaxis_title=None, yaxis_title=None)
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
        
        key_country_summary['报名数量'] = key_country_summary['报名数量'].replace(0, '无')
        
        def center_all_cols(df_style): # Accept df.style object
            return df_style.set_properties(**{
                'text-align': 'center'
            }).set_table_styles([{
                'selector': 'th',
                'props': [('text-align', 'center')]
            }])
        
        st.dataframe(center_all_cols(key_country_summary.set_index('国家').style))
        
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
            industry_counts = df[industry_col].dropna().value_counts().reset_index()
            industry_counts.columns = ['行业', '数量']
            if not industry_counts.empty:
                fig_industry = go.Figure(data=[go.Bar(
                    x=industry_counts['数量'],
                    y=industry_counts['行业'],
                    orientation='h',
                    marker_color=industry_counts['数量']
                )])
                fig_industry.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title=None, yaxis_title=None, title_text="Top 10 行业分布")
                
                # Add average line and annotation for top 10 industries
                if len(industry_counts) > 0:
                    top_10_industry_counts = industry_counts.head(10)
                    avg_value_industry = top_10_industry_counts['数量'].mean()
                    
                    fig_industry.add_shape(
                        type="line",
                        x0=avg_value_industry,
                        x1=avg_value_industry,
                        y0=-0.5, # Adjusted for horizontal bar chart (y-axis for categories)
                        y1=min(9.5, len(top_10_industry_counts) - 0.5), # Ensure y1 doesn't exceed bounds
                        line=dict(color="red", dash="dash")
                    )
                    fig_industry.add_annotation(
                        x=avg_value_industry,
                        y=min(9.5, len(top_10_industry_counts) - 0.5),
                        text=f"平均值: {avg_value_industry:.1f}",
                        showarrow=False,
                        font=dict(color="red"),
                        xanchor="left",
                        yshift=10
                    )
                st.plotly_chart(fig_industry, use_container_width=True)
            else:
                st.info(f"缺少字段：'{industry_col}' 的数据。")
        else:
            st.warning(f"缺少字段：'{industry_col}'，无法显示行业分析。")

    with col2:
        # Note: This was 'What stage is your company currently in?' originally, now renamed to '公司发展阶段' globally
        if '公司发展阶段' in df.columns:
            stage_counts_bar = df['公司发展阶段'].dropna().value_counts().reset_index()
            stage_counts_bar.columns = ['发展阶段', '数量']
            if not stage_counts_bar.empty:
                fig_stage_bar = go.Figure(data=[go.Bar(
                    x=stage_counts_bar['发展阶段'],
                    y=stage_counts_bar['数量'],
                    marker_color=stage_counts_bar['数量']
                )])
                fig_stage_bar.update_layout(title_text="公司发展阶段分布", xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig_stage_bar, use_container_width=True)
            else:
                st.info("没有发展阶段数据，无法生成图表。")
        else:
            st.warning("缺少字段：'公司发展阶段'，无法显示发展阶段分析。")

    st.markdown("---")

    # ----------------------------
    # 5. 公司类型与产品、发展阶段、营收、团队规模分析
    # (Arranged into two columns per row)
    # ----------------------------
    st.header("💼 公司运营状况分析") # More general header

    # Row 1: Company Type and Product Type
    col1_row1, col2_row1 = st.columns(2)

    with col1_row1:
        st.subheader("公司类型分布") # More specific subheader
        company_type_col = '公司类型' # Using the globally renamed column
        if company_type_col in df.columns:
            company_type_counts = df[company_type_col].dropna().value_counts().reset_index()
            company_type_counts.columns = ['公司类型', '数量']
            if not company_type_counts.empty:
                fig_company_type = go.Figure(data=[go.Pie(
                    labels=company_type_counts['公司类型'],
                    values=company_type_counts['数量'],
                    textinfo='percent+label',
                    hole=0.3
                )])
                fig_company_type.update_layout(title_text="公司类型分布")
                st.plotly_chart(fig_company_type, use_container_width=True)
            else:
                st.info(f"字段 '{company_type_col}' 没有有效数据。")
        else:
            st.warning(f"缺少字段：'{company_type_col}'，无法显示公司类型分析。")

    with col2_row1:
        st.subheader("产品类型统计") # More specific subheader
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
                # Robustly count 'Yes' values (case-insensitive) after dropping NA
                product_data[display_name] = df[original_col_name].dropna().astype(str).str.lower().eq('yes').sum()
        
        if product_data:
            product_df = pd.DataFrame(list(product_data.items()), columns=['产品类型', '数量'])
            # Filter out types with 0 count for cleaner charts
            product_df = product_df[product_df['数量'] > 0]
            if not product_df.empty:
                fig_product_type = go.Figure(data=[go.Bar(
                    x=product_df['数量'],
                    y=product_df['产品类型'],
                    orientation='h',
                    marker_color=product_df['数量']
                )])
                fig_product_type.update_layout(yaxis={'categoryorder':'total ascending'}, title_text="产品类型统计", xaxis_title="数量", yaxis_title="产品类型")
                st.plotly_chart(fig_product_type, use_container_width=True)
            else:
                st.info("没有可用的产品类型数据。")
        else:
            st.warning("缺少产品类型相关字段，无法显示产品类型分析。")

    st.markdown("---") # Separator between chart rows

    # Row 2: Development Stage and Revenue
    col1_row2, col2_row2 = st.columns(2)

    with col1_row2:
        st.subheader("📈 发展阶段分析：企业当前所处的发展阶段")
        if '公司发展阶段' in df.columns:
            stage_counts_pie = df['公司发展阶段'].dropna().value_counts().reset_index()
            stage_counts_pie.columns = ['发展阶段', '数量']
            if not stage_counts_pie.empty:
                fig_stage_pie = go.Figure(data=[go.Pie(
                    labels=stage_counts_pie['发展阶段'],
                    values=stage_counts_pie['数量'],
                    textinfo='percent+label',
                    hole=0.3
                )])
                fig_stage_pie.update_layout(title_text="企业发展阶段分布")
                st.plotly_chart(fig_stage_pie, use_container_width=True)
            else:
                st.info("无发展阶段数据，无法生成图表。")
        else:
            st.warning("缺少字段：'公司发展阶段'。")

    with col2_row2:
        st.subheader("💰 营收状况分析：企业年度营收情况分布")
        if '公司营收' in df.columns:
            revenue_counts = df['公司营收'].dropna().value_counts().reset_index()
            revenue_counts.columns = ['营收区间', '数量']
            if not revenue_counts.empty:
                fig_revenue = go.Figure(data=[go.Bar(
                    x=revenue_counts['营收区间'],
                    y=revenue_counts['数量'],
                    text=[str(x) for x in revenue_counts['数量']], # Convert numbers to strings for text
                    textposition='auto', # Show text labels automatically
                    marker_color=revenue_counts['数量'] # Color based on count
                )])
                fig_revenue.update_layout(xaxis_title="营收区间", yaxis_title="数量", title_text="企业年度营收分布")
                st.plotly_chart(fig_revenue, use_container_width=True)
            else:
                st.info("无营收数据，无法生成图表。")
        else:
            st.warning("缺少字段：'公司营收'。")
            
    st.markdown("---") # Separator between chart rows

    # Row 3: Team Size
    st.subheader("👥 团队规模分析：企业团队规模分布情况")
    if '团队规模' in df.columns:
        team_counts = df['团队规模'].dropna().value_counts().reset_index()
        team_counts.columns = ['团队规模', '数量']
        if not team_counts.empty:
            fig_team = go.Figure(data=[go.Pie(
                labels=team_counts['团队规模'],
                values=team_counts['数量'],
                textinfo='percent+label',
                hole=0.3
            )])
            fig_team.update_layout(title_text="企业团队规模分布")
            st.plotly_chart(fig_team, use_container_width=True)
        else:
            st.info("无团队规模数据，无法生成图表。")
    else:
        st.warning("缺少字段：'团队规模'。")
        
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
            fig_alibaba = go.Figure(data=[go.Pie(
                labels=alibaba_account_counts['是否有 Alibaba.com 账号'],
                values=alibaba_account_counts['数量'],
                textinfo='percent+label',
                hole=0.3
            )])
            fig_alibaba.update_layout(title_text="是否有 Alibaba.com 账号")
            st.plotly_chart(fig_alibaba, use_container_width=True)
        else:
            st.info(f"字段 '{alibaba_account_col}' 没有有效数据。")
    else:
        st.warning(f"缺少字段：'{alibaba_account_col}'，无法显示 Alibaba.com 账号分析。")
    
    feedback_content_col = 'Do you have any feedback for Alibaba.com?'
    if feedback_content_col in df.columns:
        st.subheader("部分用户反馈内容")
        sample_feedback_df = df[feedback_content_col].dropna()
        if not sample_feedback_df.empty:
            samples = sample_feedback_df.sample(min(5, len(sample_feedback_df))).tolist()
            for i, fb in enumerate(samples):
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
