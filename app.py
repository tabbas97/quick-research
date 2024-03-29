import streamlit as st
from data.local_chroma_connection import LocalChromaConnection
import chroma_wrapper
import json

searcher = chroma_wrapper.ChromaSearchWrapper('arxiv-research-paper')

# st.session_state["results"] = st.container()

def format_authors(authors):
    authors = json.loads(authors)
    print(authors)
    out = ",".join([" ".join(author) for author in authors])
    return out

def run_search():
    
    res_container : st = st.session_state["results"]
    
    query = st.session_state.get('query')
    query_results = searcher.run_query(query)
    reranked_results = searcher.rerank_queries(query, query_results)
    for result in reranked_results:
        print(result.keys())
        print(result['meta'].keys())
        with res_container.expander(result['meta']['title']):
            text_tab, meta_tab = st.tabs(["Abstract", "Metadata"])
            with meta_tab:
                st.write("ID : ", result['meta']['id'])
                st.write("Link : ", "https://arxiv.org/abs/" + result['meta']['id'])
                st.write("Authors : ", format_authors(result['meta']['authors']))
                st.write("Last Updated", result['meta']['last_updated'])
                st.write("License", result['meta']['license'])
            with text_tab:
                st.write(result['text'])
                
    st.session_state["results"] = res_container        
    

st.title('QuickResearch')
user_input = st.text_input('Enter your research topic here:')

if 'query' not in st.session_state:
    st.session_state.query = ""
    
st.session_state.query = user_input

st.button(
    'Search',
    on_click=run_search
    )

st.text('Results will appear below:')
if 'results' not in st.session_state:
    st.session_state["results"] = st.empty()