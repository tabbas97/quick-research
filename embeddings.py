from sentence_transformers import SentenceTransformer

class EmbeddingsGenerator:
    def __init__(self, model_name = "all-distilroberta-v1"):
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, sentences):
        return self.model.encode(sentences)
    
if __name__ == "__main__":
    eg = EmbeddingsGenerator()
    sentences = [
        "This is a sample sentence",
        "This is another sample sentence"
    ]
    embeddings = eg.get_embeddings(sentences)