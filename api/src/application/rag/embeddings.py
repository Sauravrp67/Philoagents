from langchain_huggingface import HuggingFaceEmbeddings
import torch

EmbeddingsModel = HuggingFaceEmbeddings

def get_embedding_model(
        model_id: str,
        device: str = "cpu"
) -> EmbeddingsModel:
    """Gets an instance of a HuggingFace embedding model.

    Args:
        model_id (str): The ID/name of the HuggingFace embedding model to use
        device (str): The compute device to run the model on (e.g. "cpu", "cuda").
            Defaults to "cpu"

    Returns:
        EmbeddingsModel: A configured HuggingFace embeddings model instance
    """

    return get_huggingface_embedding_model(model_id,device)


def get_huggingface_embedding_model(model_id:str,device:str) -> HuggingFaceEmbeddings:
    """Gets a HuggingFace embedding model instance.

    Args:
        model_id (str): The ID/name of the HuggingFace embedding model to use
        device (str): The compute device to run the model on (e.g. "cpu", "cuda")

    Returns:
        HuggingFaceEmbeddings: A configured HuggingFace embeddings model instance
            with remote code trust enabled and embedding normalization disabled
    """
    
    return HuggingFaceEmbeddings(
        model_name = model_id,
        model_kwargs = {"device":device,"trust_remote_code":True},
        encode_kwargs = {"normalize_embeddings":False},
        show_progress = True
    ) 

if __name__ == "__main__":
    sentence1 = "I am vegan"
    sentence2 = "I am not vegan"

    embedding = get_huggingface_embedding_model(
        model_id='sentence-transformers/all-MiniLM-L6-v2',
        device="cuda"
    )

    em1 = embedding.embed_documents([sentence1])
    em2 = embedding.embed_documents([sentence2])

    em1_tensor = torch.tensor(em1)
    em2_tensor = torch.tensor(em2)

    cos = torch.nn.CosineSimilarity(dim = 1,eps = 1e-8)

    score = cos(em1_tensor,em2_tensor)

    print(score)