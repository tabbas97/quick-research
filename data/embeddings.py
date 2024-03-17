from sentence_transformers import SentenceTransformer

class EmbeddingsGenerator:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, sentences):
        return self.model.encode(sentences)