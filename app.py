import streamlit as st
from data.local_chroma_connection import LocalChromaConnection
from data.embeddings import EmbeddingsGenerator
import arxiv_retrieve

def run_search(input_query: str):
    coll = LocalChromaConnection.get_collection('arxiv-research-paper')
    search_results = coll.query(
        query_embeddings=EmbeddingsGenerator.get_embeddings([input_query]),
        n_results=5
    )
    
    # Get the respective elements from streamlit and update them

st.title('QuickResearch')
st.text_input('Enter your research topic here:')

st.text('Results will appear here')







