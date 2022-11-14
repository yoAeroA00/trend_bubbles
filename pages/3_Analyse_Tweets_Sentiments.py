from Home import css, update_analysis, run_analysis, update_sentiment_frontend
import pandas
import streamlit as st

pandas.options.plotting.backend = 'hvplot'

st.set_page_config(
    page_title='Custom Analysis',
    page_icon='🎈',
    layout='wide',
    initial_sidebar_state='expanded'
)
css()

st.sidebar.title('Tweets Analysis')
analysis_folder = 'custom_analysis'
graph_height = '900px'
sentiment_charts_limit = 60

with st.sidebar.form('Request Realtime Data'):
    st.subheader('Request Realtime Data')
    query = st.text_input('Query', placeholder='keywords #Hashtags @mentions etc.', help='Please! Read (https://developer.twitter.com/en/docs/twitter-api/enterprise/search-api/overview).')
    query_type = st.radio(label='Sort by', options=['Popular', 'Mixed', 'Recent'], index=1, horizontal=True, help='Type of Tweets to Pull.')
    query_tweets = st.slider('Tweets Limit', 1, 1000, 200, help='Maximum No. of Tweets to Pull for given Query.')
    with st.expander('Depth Options', expanded=False):
        st.info('ℹ️ Use with Caution! May Overload the Browser.')
        query_depth = st.number_input('Depth', 0, 5, 0, 1, help='Depth at which to look for Hidden Information in Tweets.')
        query_depth_factor = st.number_input('Depth Factor', 1.0, value=2.0, step=0.01, help='Value to Factor with Tweets / Trend value for each added Depth.')
    form_row1col1, form_row1col2 = st.columns([1, 2])
    with form_row1col1:
        analyse = st.form_submit_button('Request')
    with form_row1col2:
        if analyse:
            if len(query) > 0:
                update_analysis(pandas.Series([query]), query_type, 1, query_tweets, int(query_depth), query_depth_factor, analysis_folder)
                run_analysis(graph_height, analysis_folder)
                st.success('✅ Database and Analysis Updated !')
            else:
                st.error('Please! Enter a Query ...', icon='🚨')

with st.spinner('Loading Analysis from Database ...'):
    update_sentiment_frontend(graph_height, sentiment_charts_limit, analysis_folder)
