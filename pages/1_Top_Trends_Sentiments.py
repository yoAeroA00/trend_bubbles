from Home import css, get_trends, update_analysis, run_analysis, update_sentiment_frontend
import json
import pandas
import streamlit as st

with open(r'data/woeids.json', 'r', encoding='utf-8') as woeids_file:
    woeids = json.load(woeids_file)

pandas.options.plotting.backend = 'hvplot'

st.set_page_config(
    page_title='Trends Analysis',
    page_icon='🎈',
    layout='wide',
    initial_sidebar_state='expanded'
)
css()

st.sidebar.title('Trends Analysis')
analysis_folder = 'trending_analysis'
graph_height = '900px'
sentiment_charts_limit = 60

with st.sidebar.form('Request Realtime Data'):
    st.subheader('Request Realtime Data')
    trends_location = st.selectbox(label='Location', options=woeids, index=0, help='Location Anchored for Analysis.')
    trends_limit = st.slider('Trends', 5, 50, 10, help='Maximum No. of Top Trending Trends for Analysis.')
    tweets_limit = st.slider('Tweets / Trend', 1, 100, 20, help='Maximum No. of Tweets to Pull per Trend (Also an Initial Point for Depth Feature).')
    trends_type = st.radio(label='Sort Tweets by', options=['Popular', 'Mixed', 'Recent'], index=1, horizontal=True, help='Type of Tweets to Pull.')
    with st.expander('Depth Options', expanded=False):
        st.info('ℹ️ Use with Caution! May Overload the Browser.')
        query_depth = st.number_input('Depth', 0, 5, 0, 1, help='Depth at which to look for Hidden Information in Tweets.')
        query_depth_factor = st.number_input('Depth Factor', 1.0, value=2.0, step=0.01, help='Value to Factor with Tweets / Trend value for each added Depth.')
    form_row1col1, form_row1col2 = st.columns([1, 2])
    with form_row1col1:
        analyse = st.form_submit_button('Request')
    with form_row1col2:
        if analyse:
            update_analysis(get_trends(woeids[trends_location]), trends_type, trends_limit, tweets_limit, int(query_depth), query_depth_factor, analysis_folder)
            run_analysis(graph_height, analysis_folder)
            st.success('✅ Database and Analysis Updated !')

with st.spinner('Loading Analysis from Database ...'):
    update_sentiment_frontend(graph_height, sentiment_charts_limit, analysis_folder)
