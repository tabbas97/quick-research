from datasets import load_dataset
import json

from local_chroma_connection import LocalChromaConnection
from embeddings import EmbeddingsGenerator

coll = LocalChromaConnection.create_collection('arxiv-research-paper', get_or_create = True)

dataset = load_dataset("Falah/arxiv-research-paper")

# all-distilroberta-v1 is a pre-trained model from sentence-transformers
# It has slightly better performance than the default model in chromaDB - all-MiniLM-L6-v2
embed_generator = EmbeddingsGenerator('all-distilroberta-v1')

print(dataset)

def prepareData(batch):
    
    ids = batch['id']
    titles = batch['title']
    licenses = batch['license']
    categories = batch['categories']
    authors = [json.dumps(per_article_author) for per_article_author in batch['authors_parsed']]
    
    last_updated = [per_article[-1]["created"] for per_article in batch["versions"]]
    
    # Primary reason to run batched_processing
    # This is faster than running the embeddings for each abstract one by one - GPU parallelism
    embeddings = embed_generator.get_embeddings(batch['abstract'])
    
    return {
        "id": ids,
        "title": titles,
        "license": licenses,
        "categories": categories,
        "authors": authors,
        "last_updated": last_updated,
        "abstract" :batch['abstract'],
        "id" :ids,
        "embeddings" :embeddings
        }

sampled_dataset = dataset['train'].shuffle(seed=42).select(range(1000))
batch_out = sampled_dataset.map(
    prepareData,
    batched=True,
    batch_size=16
    )
print(batch_out)

# Transform the metadata from columns to list of dictionaries
metadatas = [
    {
        "id": batch_out['id'][i],
        "title": batch_out['title'][i],
        "license": batch_out['license'][i] if batch_out['license'][i] else "N/A",
        "categories": batch_out['categories'][i],
        "authors": batch_out['authors'][i],
        "last_updated": batch_out['last_updated'][i]
    } for i in range(len(batch_out['id']))
]

coll.add(
    documents=batch_out['abstract'],
    embeddings=batch_out['embeddings'],
    metadatas=metadatas,
    ids=batch_out['id']
)