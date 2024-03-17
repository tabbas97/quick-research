from datasets import load_dataset
import json

from local_chroma_connection import LocalChromaConnection
from embeddings import EmbeddingsGenerator

if not LocalChromaConnection.get_collection('arxiv-research-paper'):
    coll = LocalChromaConnection.create_collection('arxiv-research-paper')
else:
    coll = LocalChromaConnection.get_collection('arxiv-research-paper')

dataset = load_dataset("Falah/arxiv-research-paper")

print(dataset)

embed_generator = EmbeddingsGenerator('all-distilroberta-v1')

def prepareDataOneSample(example):
    metadata = {
        "id": example['id'],
        "title": example['title'],
        "license": example['license'] if example['license'] else "N/A",
        "categories": example['categories'],
        "authors": json.dumps(example['authors_parsed']),
        "last_updated": example["versions"][-1]["created"]
    }
    abstract = example['abstract']

    # return metadata, abstract, example['id']
    return {
        "metadata" :metadata, 
        "abstract" :abstract, 
        "id" :example['id']
        }

def prepareData(batch):
    # print(batch, "\n\n\n")
    
    ids = batch['id']
    titles = batch['title']
    licenses = batch['license']
    categories = batch['categories']
    authors = [json.dumps(per_article_author) for per_article_author in batch['authors_parsed']]
    
    last_updated = [per_article[-1]["created"] for per_article in batch["versions"]]
    # last_updated = [print(per_article) for per_article in batch]
    
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
    metadatas=metadatas,
    ids=batch_out['id'],
    # show
)