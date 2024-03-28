import streamlit as st
import pandas as pd

st.title('Trial')
st.write('Compare this snippet from data/hf_dataset_load.py:')

from ..local_chroma_connection import LocalChromaConnection
from ..embeddings import EmbeddingsGenerator

# Check if collection exists
if not LocalChromaConnection.get_collection('arxiv-research-paper'):
    raise ValueError('Collection does not exist')

coll = LocalChromaConnection.get_collection('arxiv-research-paper')

