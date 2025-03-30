from concurrent.futures import as_completed, ThreadPoolExecutor

import numpy as np
import wikipediaapi
import chromadb
from chromadb.api.types import Images
from chromadb import Documents
from typing import Union, TypeVar

from sklearn.metrics.pairwise import cosine_similarity
import os
import nltk
from transformers import GPT2TokenizerFast

from embedding.embeddingModel import getEmbeddings
from treeQA_Config import PersistentClient_Path, chroma_collection_name
from treeQA.tree_class.embeddingModels import treeQAEmbeddings

# 初始化 tokenizer 和下载 nltk 句子分词器
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


# 定义类型
Embeddable = Union[Documents, Images]
D = TypeVar("D", bound=Embeddable, contravariant=True)

# 初始化 Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia('MyProject', 'en')

# 清除代理环境变量
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
# 初始化 Chroma 客户端
client = chromadb.PersistentClient(path=PersistentClient_Path)
embedding_function = treeQAEmbeddings()
collection = client.get_or_create_collection(name=chroma_collection_name, embedding_function=embedding_function,metadata={"hnsw:space": "cosine"})

def get_article_sections(title):
    """
    获取维基百科文章内容并按章节提取。
    """

    page = wiki_wiki.page(title)
    if not page.exists():
        print(f"Article '{title}' does not exist.")
        return []

    sections = []

    main_content = page.summary.strip()
    if main_content:
        sections.append({
            "title": "Introduction",
            "content": main_content
        })

    def extract_sections(section, prefix='', sections=None):
        section_title = f"{prefix}{section.title}"
        if "References" in section_title or "External links" in section_title:
            return

        content = section.text.strip()
        if content:
            sections.append({
                "title": section_title,
                "content": content
            })

        for subsection in section.sections:
            extract_sections(subsection, prefix=section_title + " > ", sections=sections)

    for section in page.sections:
        extract_sections(section, sections=sections)

    return sections


def split_text_by_tokens(section, max_tokens=100):
    """
    按 token 数量分割章节内容，确保每个块为完整句子，且不超过 max_tokens。
    """
    title = section['title']
    content = section['content']
    sentences = nltk.sent_tokenize(content)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence)
        #print(f"########sentences:#########\n{sentence}")
        #print(f"########sentence_tokens:#########\n{sentence_tokens}")
        sentence_length = len(sentence_tokens)

        if current_length + sentence_length > max_tokens:
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

            current_chunk = []
            current_length = 0

        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(chunk_text)

    return chunks


def is_exists(article_title):
    """
    判断文章是否已经存在于数据库中。
    """
    try:
        result = collection.get(ids=[f"{article_title}_0_0"])
        return bool(result['ids'])
    except Exception as e:
        print(f"Error checking existence: {e}")
        return False


def embed_and_store(sections, article_title):
    all_documents = []
    all_metadatas = []
    all_ids = []

    for idx, section in enumerate(sections):
        split_texts = split_text_by_tokens(section, max_tokens=100)
        for part_idx, part_text in enumerate(split_texts):
            doc_id = f"{article_title}_{idx}_{part_idx}"
            all_documents.append(part_text)
            all_metadatas.append({
                "article_title": article_title,
                "title": section['title'],
                "content": part_text
            })
            all_ids.append(doc_id)

    batch_size = 100
    total_batches = (len(all_documents) + batch_size - 1) // batch_size

    batches = []
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        batch_documents = all_documents[start:end]
        batches.append(batch_documents)


    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(getEmbeddings, batch): batch for batch in batches}
        embeddings_list = []
        for future in as_completed(futures):
            batch = futures[future]
            try:
                embeddings = future.result()
                embeddings_list.extend(embeddings)
            except Exception as e:
                print(f"Error generating embeddings for batch: {e}")

    # Now, insert all data into the collection in batches
    embeddings_index = 0
    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        batch_documents = all_documents[start:end]
        batch_metadatas = all_metadatas[start:end]
        batch_ids = all_ids[start:end]
        batch_embeddings = embeddings_list[embeddings_index:embeddings_index + len(batch_documents)]
        embeddings_index += len(batch_documents)

        try:
            collection.add(
                documents=batch_documents,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            print(f"{article_title} batch {batch_idx + 1} of {total_batches} successfully embedded and stored.")
        except Exception as e:
            print(f"Error inserting batch {batch_idx + 1}: {e}")


def query_article(query, top_k,article_name):
    """
    根据查询语句从数据库中检索相关内容。
    """
    embeddings = getEmbeddings([query])
    results = collection.query(
        query_embeddings=embeddings,
        n_results=top_k,
        include=['metadatas', 'distances' ],
        where={"article_title": {"$eq": article_name}}
    )
    query_results = []
    if 'distances' in results:
        for id, metadata, distance in zip(results['ids'][0], results['metadatas'][0], results['distances'][0]):
            if distance > 0.3:  # 过滤距离大于 0.3 的结果
                query_results.append({
                    "id": id,
                    "title": metadata['title'],
                    "content": metadata['content']
                })
    return query_results


def getWikipediaResultByNV(article_name, query , top_k=3):
    """
    获取维基百科文章结果，如果文章不存在，则进行嵌入操作。
    """
    if article_name:
        print(article_name)
        if is_exists(article_name):
            print("Article already exists in the database.")
        else:
            sections = get_article_sections(article_name)
            if sections:
                print(f"Extracted {len(sections)} sections from '{article_name}'.")
                embed_and_store(sections, article_name)

        results = query_article(query, top_k, article_name)
        return results
    return None



def embed_and_query_direct(article_name, query, top_k):
    if article_name:
        print(f"Find article {article_name}.")
    # 1. Fetch article
    sections = get_article_sections(article_name)
    if not sections:
        print(f"Article '{article_name}' not found or has no content.")
        return []

    # 2. Split into chunks
    all_chunks_data = []
    all_chunk_texts = []
    #print(f"Splitting article '{article_name}' into chunks...")
    for idx, section in enumerate(sections):
        split_texts = split_text_by_tokens(section, max_tokens=100) # Use existing function
        for part_idx, part_text in enumerate(split_texts):
            all_chunks_data.append({
                "article_title": article_name,
                "title": section['title'],
                "content": part_text,
                # Optional: Add an ID if needed later, though not strictly necessary for direct query
                "id": f"{article_name}_{idx}_{part_idx}"
            })
            all_chunk_texts.append(part_text)
    #print(f"Generated {len(all_chunk_texts)} chunks.")

    if not all_chunk_texts:
        print("No text chunks generated.")
        return []

    # 3. Embed chunks and query
    chunk_embeddings = getEmbeddings(all_chunk_texts) # Adapt if using class methods
    query_embedding = getEmbeddings([query])[0]       # Adapt if using class methods

    if not chunk_embeddings or query_embedding is None:
         print("Failed to generate embeddings.")
         return []

    # Ensure embeddings are in the correct format for cosine_similarity (e.g., 2D numpy arrays)
    chunk_embeddings_np = np.array(chunk_embeddings)
    query_embedding_np = np.array(query_embedding).reshape(1, -1)

    # 4. Calculate cosine similarities
    #print("Calculating similarities...")
    similarities = cosine_similarity(query_embedding_np, chunk_embeddings_np).flatten()

    # 5. Get top-k results
    # Get indices sorted by similarity (highest first)
    sorted_indices = np.argsort(similarities)[::-1]

    # Select top k indices
    top_k_indices = sorted_indices[:top_k]

    # 6. Format results
    #print(f"Found {len(top_k_indices)} relevant chunks.")
    results = []
    for i in top_k_indices:
        results.append({
            "id": all_chunks_data[i]["id"],
            "title": all_chunks_data[i]["title"],
            "content": all_chunks_data[i]["content"],
            "similarity": float(similarities[i]) # Add similarity score
        })

    return results

# New main function for direct query
def getWikipediaResultDirect(article_name, query, top_k=3):
    """
    Fetches, embeds, and queries a Wikipedia article directly without storing in DB.
    """
    if not article_name:
        print("Article name is required.")
        return None

    # Use the globally defined embedding_function


    results = embed_and_query_direct(article_name, query, top_k) # Pass the function/object
    return results


if __name__ == "__main__":
    print(getWikipediaResultByNV("West Germanic languages", "Which countries speak West Germanic languages?",3))

    print(getWikipediaResultDirect("West Germanic languages", "Which countries speak West Germanic languages?",3))