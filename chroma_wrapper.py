from local_chroma_connection import LocalChromaConnection
from embeddings import EmbeddingsGenerator
from embed_reranker.rerank_queries import ReRanker

class ChromaSearchWrapper():
    def __init__(self, collection_name: str):
        self.coll = LocalChromaConnection.get_collection(collection_name)
        self.emb = EmbeddingsGenerator()
        self.ranker = ReRanker()
    
    def split_query_results(self, query_results: dict) -> list[dict]:
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
    
    def run_query(
        self,
        query_text: str,
        n_results: int = 10,
        ):
        
        query_embeds = self.emb.get_embeddings([query_text])
        
        query_results = self.coll.query(
            query_embeddings = query_embeds,
            n_results = 50 if n_results > 50 else n_results
        )
        
        split_res = self.split_query_results(query_results)
        
        return split_res

    def rerank_queries(self, query_text, query_results):
        
        reranked_results = self.ranker.rerank(
            query_text, 
            query_results
            )
        
        return reranked_results
    
if __name__ == "__main__":
    wrapper = ChromaSearchWrapper('arxiv-research-paper')
    
    query_text = "balancing the magnetic field advection"
    
    query_results = wrapper.run_query(query_text)
    
    reranked_results = wrapper.rerank_queries(query_text, query_results)
    
    print(reranked_results)