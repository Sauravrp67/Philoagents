from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
import tiktoken

Splitter = RecursiveCharacterTextSplitter

tokenizer = tiktoken.get_encoding("o200k_harmony")

def get_splitter(chunk_size:int) -> Splitter:
    """
    Returns a token-based Text Splitter with overlap

    Args: 
        chunk_size (int): Number of tokens for each text chunk.

    Returns:
        Splitter: A configured text splitter instanc that splits text into overlapping chunks based on tokens
    """


    chunk_overlap = int(0.15 * chunk_size)

    logger.info(
        f"Getting Splitter with chunk_size={chunk_size} and chunk_overlap={chunk_overlap}"
    )

    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name = "o200k_harmony",
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

def count_token(text:str):
    return len(tokenizer.encode(text))

def get_avg_token(texts:list,count_fn) -> float:
    token_count = [count_fn(text.page_content) if texts[0].__class__.__name__ == "Document"
                   else count_fn(text)
                   for text in texts]
    return sum(token_count) / len(token_count)

def get_avg_char(texts:list) -> float:
    char_count = [len(text.page_content) if texts[0].__class__.__name__ == "Document"
                  else len(text)
                  for text in texts]
    return sum(char_count) / len(char_count)

if __name__ == "__main__":
    splitter = get_splitter(500)
    with open('/home/uwu/Personal_Projects/PhiloAgents/api/notebooks/llm_wiki.txt','r') as f:
        content = f.read()

    texts = splitter.split_text(content)
    avg_char_len = get_avg_char(texts)
    avg_token_len = get_avg_token(texts,count_token)

    print(f"Average Character Length per chunk: {avg_char_len}")
    print(f"Average Token Length per chunk: {avg_token_len}\n")

    for i,text in enumerate(texts[:3]):
        print(f"--- Chunk {i} ---")
        print(text)
        print("---------------\n")

