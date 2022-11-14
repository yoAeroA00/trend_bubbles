from collections import Counter
import community
from datetime import datetime
import holoviews
from io import BytesIO
import itertools
import json
import matplotlib.pyplot
import networkx
import pandas
from pyvis.network import Network
import streamlit as st
import streamlit.components.v1 as components
from tqdm.auto import tqdm
from transformers import pipeline
import tweepy
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

with open(r'data/.twitter_api_keys.json', 'r') as env_file:
    env = json.load(env_file)

dt = str(datetime.now())[:-7].replace(' ', '_').replace(':', '-')

model_path = r'data/sentiment_model'
try: sentiment = pipeline('sentiment-analysis', model=model_path, tokenizer=model_path)
except:
    st.warning('⚠️ WARNING: Failed to Load Sentiment Model, Please! Press <R> key on keyboard to reload Dashboard.')
    st.stop()

auth = tweepy.OAuth2BearerHandler(env['bearer_token'])
API = tweepy.API(auth, cache=tweepy.cache.MemoryCache(), parser=tweepy.parsers.ModelParser(), retry_count=5, retry_delay=10, retry_errors=set([401, 404, 500, 503, 10054]), wait_on_rate_limit=True)

searchdf_cols = ['query', 'created_at_date', 'created_at_time', 'id_str', 'text', 'sentiment', 'sentiment_score', 'hashtags', 'symbols',
                'user_mentions', 'result_type', 'in_reply_to_status_id_str', 'in_reply_to_user_id_str', 'in_reply_to_screen_name', 'user_id_str',
                'user_screen_name', 'user_followers_count', 'user_friends_count', 'user_favourites_count', 'user_verified', 'user_statuses_count',
                'user_profile_image_url_https', 'retweet_user_screen_name', 'is_quote_status', 'retweet_count', 'favorite_count']
sentiment_color = {'Positive' : '#00FF00', 'Neutral' : 'gray', 'Negative' : '#ff0000'}
pyviz_config = r'const options = {"nodes":{"borderWidthSelected":6,"opacity":0.6,"font":{"size":12,"face":"verdana"},"shadow":{"enabled":true}},"edges":{"arrowStrikethrough":false,"dashes":true,"labelHighlightBold":false,"scaling":{"label":false},"selfReference":{"renderBehindTheNode":false}},"interaction":{"multiselect":true,"navigationButtons":true,"hoverConnectedEdges":false},"manipulation":{"enabled":true}}'

def css():
    st.markdown('''
        <style>
            [data-testid="stSidebarNav"]::before {
                content: "Trend Bubbles";
                margin-left: 50px;
                margin-top: 10px;
                font-size: 40px;
                position: relative;
                top: 50px;
            }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
        ''', unsafe_allow_html=True,
    )

def get_trends(woeid):
    df = pandas.Series()
    try: df = pandas.json_normalize(API.get_place_trends(id=woeid), 'trends')['name']
    except:
        st.error('🚨 ERROR! No Internet Connection, Please! Try Again.')
        st.stop()
    return df

## Need to be fixed
def search_tweets(_query, _qtype, _tlimit):
    pp = 0
    pp_itr = 100 // (_tlimit + 1)
    pp_bar = st.progress(pp)
    try: search = tweepy.Cursor(API.search_tweets, q=_query, lang='en', result_type=_qtype, include_entities=True).items(_tlimit)
    except:
        st.error('🚨 ERROR! No Internet Connection, Please! Try Again.')
        st.stop()
    pp = pp + pp_itr
    pp_bar.progress(pp)
    search_df = pandas.DataFrame(columns=searchdf_cols)
    for item in tqdm(search, total=_tlimit, desc='Downloading Tweets', leave=False):
        pp = pp + pp_itr
        pp_bar.progress(pp)
        created_at = pandas.to_datetime(item.created_at)
        senti = sentiment(item.text)[0]
        try: rt_user_screen_name = item.retweet_status.user['screen_name']
        except: rt_user_screen_name = None
        search_df.loc[len(search_df.index)] = [_query, created_at.date(), created_at.time(), item.id_str, item.text, senti['label'], senti['score'],
                                 ['#' + hashtag['text'] for hashtag in item.entities['hashtags'] if hashtag['text'] != _query.strip(' ').lstrip('#')],
                                 [symbol['text'] for symbol in item.entities['symbols']],
                                 [[user['id'], user['screen_name'], user['name']] for user in item.entities['user_mentions']], item.metadata['result_type'],
                                 item.in_reply_to_status_id_str, item.in_reply_to_user_id_str, item.in_reply_to_screen_name, item.user.id_str, item.user.screen_name,
                                 item.user.followers_count, item.user.friends_count, item.user.favourites_count, bool(item.user.verified), item.user.statuses_count, 
                                 item.user.profile_image_url_https, rt_user_screen_name, bool(item.is_quote_status), item.retweet_count, item.favorite_count]
    pp_bar.empty()
    return search_df

# Need to be fixed
def search_tweets_depth(_query_series, _qtype, _qlimit, _tlimit, _depth, _qdepth_factor):
    queries = set(_query_series[:_qlimit])
    qlimit = len(queries)
    search_df = pandas.DataFrame(columns=searchdf_cols)
    if _depth > 0:
        dp = 0
        dp_itr = 100 // (_depth + 1)
        dp_bar = st.progress(dp)
    for depth in range(_depth + 1):
        if _depth > 0:
            dp = dp + dp_itr
            dp_bar.progress(dp)
        if qlimit > 4:
            qp = 0
            qp_itr = 100 // qlimit
            qp_bar = st.progress(qp)
        for query in tqdm(queries, total=qlimit, desc=f'Working on Queries at Depth ({_depth - 1})', leave=False):
            if qlimit > 4:
                qp = qp + qp_itr
                qp_bar.progress(qp)
            df = search_tweets(query, _qtype, int(_tlimit))
            search_df = pandas.concat([search_df, df], axis=0)
        if qlimit > 4: qp_bar.empty()
        search_df.drop_duplicates('id_str', inplace=True)
        new_queries = set()
        for hashtags in search_df['hashtags']:
            if len(hashtags) > 0:
                new_queries.update(hashtags)
        queries = new_queries - queries
        _tlimit = _tlimit * _qdepth_factor
    if _depth > 0: dp_bar.empty()
    return search_df

def df2hut(_df):
    hashtag_edges = []
    hashtag_wedges = []
    users_edges = []
    users_wedges = []
    tweets_edges = []
    tweets_wedges = []
    for index, row in _df.iterrows():
        if len(row['hashtags']) > 0:
            for edge in map(list, itertools.product(row['hashtags'], [row['query'].split(' ')[0]])):
                hashtag_edges.append(edge)
            for edge in map(list, itertools.product([row['user_screen_name']], row['hashtags'])):
                tweets_edges.append([edge, row['sentiment']])
        if len(row['user_mentions']) > 0:
            for edge in map(list, itertools.product([row['user_screen_name']], [user[1] for user in row['user_mentions']])):
                users_edges.append([edge, row['sentiment']])
        if row['in_reply_to_screen_name'] != None:
            for edge in map(list, itertools.product([row['user_screen_name']], [row['in_reply_to_screen_name']])):
                users_edges.append([edge, row['sentiment']])
        if row['retweet_user_screen_name'] != None:
            for edge in map(list, itertools.product([row['user_screen_name']], [row['retweet_user_screen_name']])):
                users_edges.append([edge, row['sentiment']])
    tmp = list(Counter([edge[0]+'|-+~|'+edge[1] for edge in hashtag_edges]).items())
    for i in tmp:
        fro, to = i[0].split('|-+~|')
        hashtag_wedges.append([fro, to, i[1], ''])
    tmp = list(Counter([edge[0][0]+'|-+~|'+edge[0][1]+'|-+~|'+edge[1] for edge in users_edges]).items())
    for i in tmp:
        fro, to, sent = i[0].split('|-+~|')
        users_wedges.append([fro, to, i[1], sent])
    tmp = list(Counter([edge[0][0]+'|-+~|'+edge[0][1]+'|-+~|'+edge[1] for edge in tweets_edges]).items())
    for i in tmp:
        fro, to, sent = i[0].split('|-+~|')
        tweets_wedges.append([fro, to, i[1], sent])
    hashtag_nodes = set([hashtag for hashtags in hashtag_edges for hashtag in hashtags])
    users_nodes = set([user for users in users_edges for user in users[0]])
    tweets_nodes = hashtag_nodes & set(['@'+user for user in users_nodes])
    return [[hashtag_nodes, hashtag_wedges], [users_nodes, users_wedges], [tweets_nodes, tweets_wedges]]

def df2excel(_df: pandas.DataFrame):
    in_memory_fp = BytesIO()
    _df.to_excel(in_memory_fp, index=False)
    in_memory_fp.seek(0, 0)
    return in_memory_fp.read()

def graph_node_attributes(_graph):
    networkx.set_node_attributes(_graph, dict(_graph.degree), 'size')
    networkx.set_node_attributes(_graph, networkx.degree_centrality(_graph), 'degree_centrality')
    networkx.set_node_attributes(_graph, networkx.betweenness_centrality(_graph), 'betweenness_centrality')
    networkx.set_node_attributes(_graph, networkx.closeness_centrality(_graph), 'closeness_centrality')
    networkx.set_node_attributes(_graph, networkx.pagerank(_graph, alpha=0.85), 'page_rank')

def graph2nodesdf(_graph):
    nodes_df = pandas.DataFrame.from_dict(dict(_graph.nodes(data=True)), orient='index')
    nodes_df.rename(columns = {'size':'degree', 'group':'louvain_community'}, inplace = True)
    return nodes_df

def get_hashtag_network(_hashtag_data, _height, _width, _bgcolor, _font_color):
    hashtag_graph = networkx.Graph()
    #hashtag_graph.add_nodes_from(_hashtag_data[0])
    for edge in _hashtag_data[1]:
        hashtag_graph.add_edge(edge[0], edge[1], weight=edge[2])
    graph_node_attributes(hashtag_graph)
    networkx.set_node_attributes(hashtag_graph, community.best_partition(hashtag_graph), 'group')
    hashtag_net = Network(height=_height, width=_width, directed=False, bgcolor=_bgcolor, font_color=_font_color)
    hashtag_net.from_nx(hashtag_graph)
    hashtag_net.set_options(pyviz_config)
    return [hashtag_net, hashtag_graph]

def get_users_network(_community, _users_data, _height, _width, _bgcolor, _font_color):
    users_graph = networkx.Graph()
    #users_graph.add_nodes_from(set(['@'+user for user in _users_data[0]]))
    for edge in [edges for edges in _users_data[1][::-1]]:
        if _community:
            users_graph.add_edge('@'+edge[0], '@'+edge[1], weight=edge[2])
        else:
            users_graph.add_edge('@'+edge[0], '@'+edge[1], weight=edge[2], color=sentiment_color[edge[3]])
    graph_node_attributes(users_graph)
    if _community:
        networkx.set_node_attributes(users_graph, community.best_partition(users_graph), 'group')
    users_net = Network(height=_height, width=_width, directed=False, bgcolor=_bgcolor, font_color=_font_color)
    users_net.from_nx(users_graph)
    users_net.set_options(pyviz_config)
    return [users_net, users_graph]

def get_tweets_network(_community, _hashtag_data, _users_data, _tweets_data, _height, _width, _bgcolor, _font_color):
    tweets_graph = networkx.Graph()
    #tweets_graph.add_nodes_from(_tweets_data[0])
    for edge in _hashtag_data[1]:
        tweets_graph.add_edge(edge[0], edge[1], weight=edge[2])
    for edge in [edges for edges in _users_data[1][::-1]]:
        if _community:
            tweets_graph.add_edge('@'+edge[0], '@'+edge[1], weight=edge[2])
        else:
            tweets_graph.add_edge('@'+edge[0], '@'+edge[1], weight=edge[2], color=sentiment_color[edge[3]])
    for edge in [edges for edges in _tweets_data[1][::-1]]:
        if _community:
            tweets_graph.add_edge('@'+edge[0], edge[1], weight=edge[2])
        else:
            tweets_graph.add_edge('@'+edge[0], edge[1], weight=edge[2], color=sentiment_color[edge[3]])
    graph_node_attributes(tweets_graph)
    if _community:
        networkx.set_node_attributes(tweets_graph, community.best_partition(tweets_graph), 'group')
    tweets_net = Network(height=_height, width=_width, directed=False, bgcolor=_bgcolor, font_color=_font_color)
    tweets_net.from_nx(tweets_graph)
    tweets_net.set_options(pyviz_config)
    return [tweets_net, tweets_graph]

def process_sentiments_df(_df):
    df = _df.groupby('sentiment').count()['sentiment_score']
    if 'Negative' not in df.index: df['Negative'] = 0
    if 'Neutral' not in df.index: df['Neutral'] = 0
    if 'Positive' not in df.index: df['Positive'] = 0
    return df.sort_index()

def fill_sentiments(_df):
    df = _df.copy()
    if 'Negative' not in df.columns: df['Negative'] = 0.0
    if 'Neutral' not in df.columns: df['Neutral'] = 0.0
    if 'Positive' not in df.columns: df['Positive'] = 0.0
    return df

def update_analysis(_queries_series, _queries_type, _queries_limit, _tweets_limit, _query_depth, _query_depth_factor, _analysis_folder):
    search_df = search_tweets_depth(_queries_series, _queries_type.lower(), _queries_limit, _tweets_limit, _query_depth, _query_depth_factor)
    if len(search_df.index) < 1:
        st.warning('⚠️ WARNING! No Tweets Found, Please! Try Again.')
        st.stop()
    search_df.to_pickle(f'data/{_analysis_folder}/search_df.pkl')

def run_analysis(_graph_height, _analysis_folder):
    with open(f'data/{_analysis_folder}/search_df.pkl', 'rb') as search_df_pkl:
        search_df = pandas.read_pickle(search_df_pkl)
    graph_width = '100%'
    sentiments_graph_bgcolor = 'whitesmoke'
    sentiments_graph_font_color = 'black'
    community_graph_bgcolor = 'whitesmoke'
    community_graph_font_color = 'black'
    hashtag_data, users_data, tweets_data = df2hut(search_df)
    hashtag_cnet, hashtag_cgraph = get_hashtag_network(hashtag_data, _graph_height, graph_width, community_graph_bgcolor, community_graph_font_color)
    users_cnet, users_cgraph = get_users_network(True, users_data, _graph_height, graph_width, community_graph_bgcolor, community_graph_font_color)
    users_net, users_graph = get_users_network(False, users_data, _graph_height, graph_width, sentiments_graph_bgcolor, sentiments_graph_font_color)
    tweets_cnet, tweets_cgraph = get_tweets_network(True, hashtag_data, users_data, tweets_data, _graph_height, graph_width, community_graph_bgcolor, community_graph_font_color)
    tweets_net, tweets_graph = get_tweets_network(False, hashtag_data, users_data, tweets_data, _graph_height, graph_width, sentiments_graph_bgcolor, sentiments_graph_font_color)
    hashtag_cnet.write_html(f'data/{_analysis_folder}/hashtag_communities.html')
    users_cnet.write_html(f'data/{_analysis_folder}/users_communities.html')
    users_net.write_html(f'data/{_analysis_folder}/users_sentiment.html')
    tweets_cnet.write_html(f'data/{_analysis_folder}/tweets_communities.html')
    tweets_net.write_html(f'data/{_analysis_folder}/tweets_sentiment.html')
    networkx.write_graphml(hashtag_cgraph, f'data/{_analysis_folder}/hashtag_communities.graphml')
    networkx.write_graphml(users_cgraph, f'data/{_analysis_folder}/users_communities.graphml')
    networkx.write_graphml(users_graph, f'data/{_analysis_folder}/users_sentiment.graphml')
    networkx.write_graphml(tweets_cgraph, f'data/{_analysis_folder}/tweets_communities.graphml')
    networkx.write_graphml(tweets_graph, f'data/{_analysis_folder}/tweets_sentiment.graphml')
    st.balloons()

def update_sentiment_frontend(_graph_height, _sentiment_charts_limit, _analysis_folder):
    st.title('Sentiment Analysis')
    with open(f'data/{_analysis_folder}/users_sentiment.html', 'r', encoding='utf-8') as users_sentiment:
        users_sentiment_html = users_sentiment.read()
    with open(f'data/{_analysis_folder}/users_sentiment.graphml', 'rb') as users_graph_gml:
        users_graph = networkx.read_graphml(users_graph_gml)
    with open(f'data/{_analysis_folder}/tweets_sentiment.html', 'r', encoding='utf-8') as tweets_sentiment:
        tweets_sentiment_html = tweets_sentiment.read()
    with open(f'data/{_analysis_folder}/tweets_sentiment.graphml', 'rb') as tweets_graph_gml:
        tweets_graph = networkx.read_graphml(tweets_graph_gml)
    with open(f'data/{_analysis_folder}/search_df.pkl', 'rb') as search_df_pkl:
        search_df = pandas.read_pickle(search_df_pkl)
    
    sentiment_tab1, sentiment_tab2, sentiment_tab3 = st.tabs(["⟳ Graph", "📈 Charts", "🗃 Data"])
    with sentiment_tab1:
        st.subheader('Users Network Graph')
        st.info('ℹ️ Edges Color represent the state of last interaction between the two users(Nodes), which is classified in three categories, Negative (RED), Neutral (GRAY), Positive (GREEN).')
        components.html(users_sentiment_html, height=int(_graph_height[:-2])+10, scrolling=True)
        st.subheader('Tweets Network Graph')
        st.info('ℹ️ Edges Color represent the state of last interaction between the two users(Nodes), which is classified in three categories, Negative (RED), Neutral (GRAY), Positive (GREEN). Additionally, An fourth (Faded BLUE) coloured Edge represent relation between two Hashtags(Nodes).')
        components.html(tweets_sentiment_html, height=int(_graph_height[:-2])+10, scrolling=True)
    with sentiment_tab2:
        st.subheader('Cumulative Sentiments Distribution')
        sentiments_df = process_sentiments_df(search_df)
        search_row1col1, search_row1col2 = st.columns(2, gap='small')
        sentiment_pfig, sentiment_pchart = matplotlib.pyplot.subplots(figsize=(4, 4))
        sentiment_pchart = matplotlib.pyplot.pie(sentiments_df, labels=sentiments_df.index, autopct='%.1f%%', colors=[sentiment_color['Negative'], sentiment_color['Neutral'], sentiment_color['Positive']], explode=(0.05, 0.01, 0.04))
        with search_row1col1:
            st.pyplot(sentiment_pfig)
        with search_row1col2:
            sentiment_bfig, sentiment_bchart = matplotlib.pyplot.subplots(figsize=(4, 4))
            sentiment_bchart = matplotlib.pyplot.bar(sentiments_df.index, sentiments_df.values, color=[sentiment_color['Negative'], sentiment_color['Neutral'], sentiment_color['Positive']])
            sentiment_bchart = matplotlib.pyplot.xlabel('Sentiment', fontsize=5)
            sentiment_bchart = matplotlib.pyplot.ylabel('Number of Tweets', fontsize=6)
            sentiment_bchart = matplotlib.pyplot.tick_params(axis='both', labelsize=5)
            st.pyplot(sentiment_bfig)
        if _analysis_folder == 'trending_analysis':
            st.caption('Most Tweeted Hashtags Sentiment')
            sentiment_bhchartdf = fill_sentiments(search_df.groupby(['query', 'sentiment'])['sentiment'].count().unstack('sentiment').fillna(0))
            sentiment_bhchartdf['sentiments_count'] = sentiment_bhchartdf['Negative'] + sentiment_bhchartdf['Neutral'] + sentiment_bhchartdf['Positive']
            sentiment_bhchart = sentiment_bhchartdf.sort_values('sentiments_count')[-_sentiment_charts_limit:][::-1].plot.bar(stacked=True, height=500, width=1400, xlabel='Users', y=['Negative', 'Neutral', 'Positive'], ylabel='Number of Tweets', color=[sentiment_color['Negative'], sentiment_color['Neutral'], sentiment_color['Positive']], legend='top_right').opts(xrotation=45, active_tools=['box_zoom'])
            st.bokeh_chart(holoviews.render(sentiment_bhchart, backend='bokeh'), use_container_width=True)
        st.caption('Most Active Users Tweets Sentiment')
        sentiment_bechartdf = search_df.copy()
        sentiment_bechartdf['user_screen_name'] = '@' + sentiment_bechartdf['user_screen_name']
        sentiment_bechartdf = fill_sentiments(sentiment_bechartdf.groupby(['user_screen_name', 'sentiment'])['sentiment'].count().unstack('sentiment').fillna(0))
        sentiment_bechartdf['edges_count'] = sentiment_bechartdf['Negative'] + sentiment_bechartdf['Neutral'] + sentiment_bechartdf['Positive']
        sentiment_bechart = sentiment_bechartdf.sort_values('edges_count')[-_sentiment_charts_limit:][::-1].plot.bar(stacked=True, height=500, width=1400, xlabel='Users', y=['Negative', 'Neutral', 'Positive'], ylabel='Number of Tweets', color=[sentiment_color['Negative'], sentiment_color['Neutral'], sentiment_color['Positive']], legend='top_right').opts(xrotation=45, active_tools=['box_zoom'])
        st.bokeh_chart(holoviews.render(sentiment_bechart, backend='bokeh'), use_container_width=True)
    with sentiment_tab3:
        st.caption('Graphs Data')
        sentiment_tab3_row1col1, sentiment_tab3_row1col2, sentiment_tab3_row1col3, sentiment_tab3_row1col4, sentiment_tab3_row1col5, sentiment_tab3_row1col6 = st.columns([2, 1, 1, 2, 1, 1])
        with sentiment_tab3_row1col1:
            st.write('Download Users Graph Data in')
        with sentiment_tab3_row1col2:
            with open(f'data/{_analysis_folder}/users_sentiment.graphml', 'rb') as users_graph_gml:
                st.download_button(label='GML Format', data=users_graph_gml, file_name=f'Users_Sentiment_Graph_{dt}.gml', mime='application/octet-stream')
        with sentiment_tab3_row1col3:
            st.download_button(label='JSON Format', data=json.dumps(networkx.node_link_data(users_graph), indent=True, ensure_ascii=False), file_name=f'Users_Sentiment_Graph_{dt}.json', mime='application/json')
        with sentiment_tab3_row1col4:
            st.write('Download Tweets Graph Data in')
        with sentiment_tab3_row1col5:
            with open(f'data/{_analysis_folder}/tweets_sentiment.graphml', 'rb') as tweets_graph_gml:
                st.download_button(label='GML Format', data=tweets_graph_gml, file_name=f'Tweets_Sentiment_Graph_{dt}.gml', mime='application/octet-stream')
        with sentiment_tab3_row1col6:
            st.download_button(label='JSON Format', data=json.dumps(networkx.node_link_data(tweets_graph), indent=True, ensure_ascii=False), file_name=f'Tweets_Sentiment_Graph_{dt}.json', mime='application/json')
        sentiment_tab3_row2col1, sentiment_tab3_row2col2 = st.columns([5, 1])
        with sentiment_tab3_row2col1:
            st.empty()
            st.caption('Charts Data')
        with sentiment_tab3_row2col2:
            st.download_button(label='Download Chart Data', data=df2excel(search_df), file_name=f'Trends_Sentiment_Data_{dt}.xlsx', mime='application/xlsx')
        st.dataframe(search_df)

def update_community_frontend(_graph_height, _hashtag_charts_limit, _users_charts_limit, _tweets_charts_limit, _analysis_folder):
    st.title('Community Analysis')
    with open(f'data/{_analysis_folder}/hashtag_communities.html', 'r', encoding='utf-8') as hashtag_communities:
        hashtag_communities_html = hashtag_communities.read()
    with open(f'data/{_analysis_folder}/hashtag_communities.graphml', 'rb') as hashtag_cgraph_gml:
        hashtag_cgraph = networkx.read_graphml(hashtag_cgraph_gml)
    with open(f'data/{_analysis_folder}/users_communities.html', 'r', encoding='utf-8') as users_communities:
        users_communities_html = users_communities.read()
    with open(f'data/{_analysis_folder}/users_communities.graphml', 'rb') as users_cgraph_gml:
        users_cgraph = networkx.read_graphml(users_cgraph_gml)
    with open(f'data/{_analysis_folder}/tweets_communities.html', 'r', encoding='utf-8') as tweets_communities:
        tweets_communities_html = tweets_communities.read()
    with open(f'data/{_analysis_folder}/tweets_communities.graphml', 'rb') as tweets_cgraph_gml:
        tweets_cgraph = networkx.read_graphml(tweets_cgraph_gml)
    hashtag_nodesdf = graph2nodesdf(hashtag_cgraph)
    users_nodesdf = graph2nodesdf(users_cgraph)
    tweets_nodesdf = graph2nodesdf(tweets_cgraph)
    
    st.subheader('Hashtag Network Graph')
    hashtag_tab1, hashtag_tab2, hashtag_tab3 = st.tabs(["⟳ Graph", "📈 Charts", "🗃 Data"])
    with hashtag_tab1:
        st.info('ℹ️ Similarly Colored Cluster of Edges and Nodes represent a Single Community.')
        components.html(hashtag_communities_html, height=int(_graph_height[:-2])+10, scrolling=True)
    with hashtag_tab2:
        st.caption('Most Important Hashtags (Page Rank)')
        st.bokeh_chart(holoviews.render(hashtag_nodesdf.sort_values('page_rank')[-_hashtag_charts_limit:][::-1].plot.bar(height=500, width=1400, x='index', xlabel='Trends', y='page_rank', ylabel='Page Rank', yformatter='%.4f').opts(xrotation=45, active_tools=['box_zoom']), backend='bokeh'), use_container_width=True)
        st.caption('Most Influencial Hashtags (Degree Centrality)')
        st.bokeh_chart(holoviews.render(hashtag_nodesdf.sort_values('degree_centrality')[-_hashtag_charts_limit:][::-1].plot.bar(height=500, width=1400, x='index', xlabel='Trends', y='degree_centrality', ylabel='Degree Centrality', yformatter='%.4f').opts(xrotation=45, active_tools=['box_zoom']), backend='bokeh'), use_container_width=True)
        st.caption('Important Gatekeepers of Information in Hashtag Network (Betweenness Centrality)')
        st.bokeh_chart(holoviews.render(hashtag_nodesdf.sort_values('betweenness_centrality')[-_hashtag_charts_limit:][::-1].plot.bar(height=500, width=1400, x='index', xlabel='Trends', y='betweenness_centrality', ylabel='Betweenness Centrality', yformatter='%.4f').opts(xrotation=45, active_tools=['box_zoom']), backend='bokeh'), use_container_width=True)
        st.caption('Broadcasters of Information in Hashtag Network (Closeness Centrality)')
        st.bokeh_chart(holoviews.render(hashtag_nodesdf.sort_values('closeness_centrality')[-_hashtag_charts_limit:][::-1].plot.bar(height=500, width=1400, x='index', xlabel='Trends', y='closeness_centrality', ylabel='Closeness Centrality', yformatter='%.3f').opts(xrotation=45, active_tools=['box_zoom']), backend='bokeh'), use_container_width=True)
    with hashtag_tab3:
        st.caption('Graph Data')
        hashtag_tab3_row1col1, hashtag_tab3_row1col2, hashtag_tab3_row1col3, hashtag_tab3_row1col4 = st.columns([2, 1, 1, 5])
        with hashtag_tab3_row1col1:
            st.write('Download Graph Data in')
        with hashtag_tab3_row1col2:
            with open(f'data/{_analysis_folder}/hashtag_communities.graphml', 'rb') as hashtag_cgraph_gml:
                st.download_button(label='GML Format', data=hashtag_cgraph_gml, file_name=f'Hashtag_Community_Graph_{dt}.gml', mime='application/octet-stream')
        with hashtag_tab3_row1col3:
            st.download_button(label='JSON Format', data=json.dumps(networkx.node_link_data(hashtag_cgraph), indent=True, ensure_ascii=False), file_name=f'Hashtag_Community_Graph_{dt}.json', mime='application/json')
        hashtag_tab3_row2col1, hashtag_tab3_row2col2 = st.columns([5, 1])
        with hashtag_tab3_row2col1:
            st.empty()
            st.caption('Charts Data')
        with hashtag_tab3_row2col2:
            st.download_button(label='Download Chart Data', data=df2excel(hashtag_nodesdf), file_name=f'Hashtag_Charts_Data_{dt}.xlsx', mime='application/xlsx')
        st.dataframe(hashtag_nodesdf)
    
    st.subheader('Users Network Graph')
    users_tab1, users_tab2, users_tab3 = st.tabs(["⟳ Graph", "📈 Charts", "🗃 Data"])
    with users_tab1:
        st.info('ℹ️ Similarly Colored Cluster of Edges and Nodes represent a Single Community.')
        components.html(users_communities_html, height=int(_graph_height[:-2])+10, scrolling=True)
    with users_tab2:
        st.caption('Most Important Users (Page Rank)')
        st.bokeh_chart(holoviews.render(users_nodesdf.sort_values('page_rank')[-_users_charts_limit:][::-1].plot.bar(height=500, width=1400, x='index', xlabel='Trends', y='page_rank', ylabel='Page Rank', yformatter='%.4f').opts(xrotation=45, active_tools=['box_zoom']), backend='bokeh'), use_container_width=True)
        st.caption('Most Influencial Users (Degree Centrality)')
        st.bokeh_chart(holoviews.render(users_nodesdf.sort_values('degree_centrality')[-_users_charts_limit:][::-1].plot.bar(height=500, width=1400, x='index', xlabel='Trends', y='degree_centrality', ylabel='Degree Centrality', yformatter='%.4f').opts(xrotation=45, active_tools=['box_zoom']), backend='bokeh'), use_container_width=True)
        st.caption('Important Gatekeepers of Information in Users Network (Betweenness Centrality)')
        st.bokeh_chart(holoviews.render(users_nodesdf.sort_values('betweenness_centrality')[-_users_charts_limit:][::-1].plot.bar(height=500, width=1400, x='index', xlabel='Trends', y='betweenness_centrality', ylabel='Betweenness Centrality', yformatter='%.4f').opts(xrotation=45, active_tools=['box_zoom']), backend='bokeh'), use_container_width=True)
        st.caption('Broadcasters of Information in Users Network (Closeness Centrality)')
        st.bokeh_chart(holoviews.render(users_nodesdf.sort_values('closeness_centrality')[-_users_charts_limit:][::-1].plot.bar(height=500, width=1400, x='index', xlabel='Trends', y='closeness_centrality', ylabel='Closeness Centrality', yformatter='%.3f').opts(xrotation=45, active_tools=['box_zoom']), backend='bokeh'), use_container_width=True)
    with users_tab3:
        st.caption('Graph Data')
        users_tab3_row1col1, users_tab3_row1col2, users_tab3_row1col3, users_tab3_row1col4 = st.columns([2, 1, 1, 5])
        with users_tab3_row1col1:
            st.write('Download Graph Data in')
        with users_tab3_row1col2:
            with open(f'data/{_analysis_folder}/users_communities.graphml', 'rb') as users_cgraph_gml:
                st.download_button(label='GML Format', data=users_cgraph_gml, file_name=f'Users_Community_Graph_{dt}.gml', mime='application/octet-stream')
        with users_tab3_row1col3:
            st.download_button(label='JSON Format', data=json.dumps(networkx.node_link_data(users_cgraph), indent=True, ensure_ascii=False), file_name=f'Users_Community_Graph_{dt}.json', mime='application/json')
        users_tab3_row2col1, users_tab3_row2col2 = st.columns([5, 1])
        with users_tab3_row2col1:
            st.empty()
            st.caption('Charts Data')
        with users_tab3_row2col2:
            st.download_button(label='Download Chart Data', data=df2excel(users_nodesdf), file_name=f'Users_Charts_Data_{dt}.xlsx', mime='application/xlsx')
        st.dataframe(users_nodesdf)
    
    st.subheader('Tweets Network Graph')
    tweets_tab1, tweets_tab2, tweets_tab3 = st.tabs(["⟳ Graph", "📈 Charts", "🗃 Data"])
    with tweets_tab1:
        st.info('ℹ️ Similarly Colored Cluster of Edges and Nodes represent a Single Community.')
        components.html(tweets_communities_html, height=int(_graph_height[:-2])+10, scrolling=True)
    with tweets_tab2:
        st.caption('Most Important Hashtags / Users (Page Rank)')
        st.bokeh_chart(holoviews.render(tweets_nodesdf.sort_values('page_rank')[-_tweets_charts_limit:][::-1].plot.bar(height=500, width=1400, x='index', xlabel='Trends', y='page_rank', ylabel='Page Rank', yformatter='%.4f').opts(xrotation=45, active_tools=['box_zoom']), backend='bokeh'), use_container_width=True)
        st.caption('Most Influencial Hashtags / Users (Degree Centrality)')
        st.bokeh_chart(holoviews.render(tweets_nodesdf.sort_values('degree_centrality')[-_tweets_charts_limit:][::-1].plot.bar(height=500, width=1400, x='index', xlabel='Trends', y='degree_centrality', ylabel='Degree Centrality', yformatter='%.4f').opts(xrotation=45, active_tools=['box_zoom']), backend='bokeh'), use_container_width=True)
        st.caption('Important Gatekeepers of Information in Tweets Network (Betweenness Centrality)')
        st.bokeh_chart(holoviews.render(tweets_nodesdf.sort_values('betweenness_centrality')[-_tweets_charts_limit:][::-1].plot.bar(height=500, width=1400, x='index', xlabel='Trends', y='betweenness_centrality', ylabel='Betweenness Centrality', yformatter='%.4f').opts(xrotation=45, active_tools=['box_zoom']), backend='bokeh'), use_container_width=True)
        st.caption('Broadcasters of Information in Tweets Network (Closeness Centrality)')
        st.bokeh_chart(holoviews.render(tweets_nodesdf.sort_values('closeness_centrality')[-_tweets_charts_limit:][::-1].plot.bar(height=500, width=1400, x='index', xlabel='Trends', y='closeness_centrality', ylabel='Closeness Centrality', yformatter='%.3f').opts(xrotation=45, active_tools=['box_zoom']), backend='bokeh'), use_container_width=True)
    with tweets_tab3:
        st.caption('Graph Data')
        tweets_tab3_row1col1, tweets_tab3_row1col2, tweets_tab3_row1col3, tweets_tab3_row1col4 = st.columns([2, 1, 1, 5])
        with tweets_tab3_row1col1:
            st.write('Download Graph Data in')
        with tweets_tab3_row1col2:
            with open(f'data/{_analysis_folder}/tweets_communities.graphml', 'rb') as tweets_cgraph_gml:
                st.download_button(label='GML Format', data=tweets_cgraph_gml, file_name=f'Tweets_Community_Graph_{dt}.gml', mime='application/octet-stream')
        with tweets_tab3_row1col3:
            st.download_button(label='JSON Format', data=json.dumps(networkx.node_link_data(tweets_cgraph), indent=True, ensure_ascii=False), file_name=f'Tweets_Community_Graph_{dt}.json', mime='application/json')
        tweets_tab3_row2col1, tweets_tab3_row2col2 = st.columns([5, 1])
        with tweets_tab3_row2col1:
            st.empty()
            st.caption('Charts Data')
        with tweets_tab3_row2col2:
            st.download_button(label='Download Chart Data', data=df2excel(tweets_nodesdf), file_name=f'Tweets_Charts_Data_{dt}.xlsx', mime='application/xlsx')
        st.dataframe(tweets_nodesdf)

if __name__ == '__main__':
    st.set_page_config(
        page_title='Trend Bubbles',
        page_icon='🎈',
        layout='centered',
        initial_sidebar_state='expanded'
    )
    css()

    st.sidebar.success('✅ Select an Analysis Above.')

    st.write('# Welcome to Trend Bubbles! 👋')

    st.markdown(
        '''
        Trend Bubbles is an Social Media Analytics Dashboard built using
        Machine Learning and Data Science Analysis Algorithms.
        **👈 Select an Analysis from the Sidebar** to start your Journey.
        ### What can I do with this App?
         - Pull Realtime Twitter Data.
         - Perform Sentiment analysis on the Data.
         - Interactive Network Graph Analysis.
         - Community Detection on Hashtags, Users, Tweets.
         - Download the Data for lateral use.
         - and much more.....
    '''
    )

    st.balloons()
