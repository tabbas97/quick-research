from flashrank import Ranker, RerankRequest

class ReRanker():
    def __init__(self) -> None:
        self.ranker = Ranker(
            model_name = "ms-marco-MiniLM-L-12-v2",
            cache_dir = "cache"
        )
        
    def format_documents(self, retrieved_results: dict) -> list[dict]:
        
        documents = []
        for retr in retrieved_results:
            documents.append({
                'id': retr['ids'],
                'text': retr['documents'],
                'meta': retr['metadatas']
            })
        return documents
        
    def rerank(self, query: str, retrieved_results: list[dict]) -> list[str]:
        
        documents = self.format_documents(retrieved_results)
        
        request = RerankRequest(
            query = query,
            passages = documents
        )
        
        results = self.ranker.rerank(request)
        return results
    
def split_query_results(query_results: dict) -> list[dict]:
    
    out = [{} for i in range(len(query_results['ids'][0]))]
    out_keys = list(query_results.keys())
    
    for key in out_keys:
        if query_results[key] is None:
            for i in range(len(out)):
                out[i][key] = None
            continue
        for i in range(len(out)):
            out[i][key] = query_results[key][0][i]
    
    return out
    
if __name__ == "__main__":
    
    from local_chroma_connection import LocalChromaConnection
    
    coll = LocalChromaConnection.get_collection('arxiv-research-paper')
    reranker = ReRanker()
    
    from embeddings import EmbeddingsGenerator
    
    emb = EmbeddingsGenerator()
    
    query_texts = ["balancing the magnetic field advection"]
    
    embedded = emb.get_embeddings(query_texts)
    
    query_res = coll.query(
        query_embeddings = embedded,
        n_results=10
    )
    
    out = split_query_results(query_res)
    
    reranked = reranker.rerank("balancing the magnetic field advection", out)
    
    print(reranked)
    for res in reranked:
        print(res['score'], res['text'])