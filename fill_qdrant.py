import argparse
import sys
from pathlib import Path
from qdrant_client.http.models import (
    PointStruct, VectorParams, Distance, PayloadSchemaType
)
from qdrant_client.http.models import TextIndexParams, TokenizerType
from qdrant_client import QdrantClient, models
import numpy as np
import pandas as pd
import json
from more_itertools import batched
from tqdm import tqdm
from openai import OpenAI

from config import LLM_NAME, ENCODER_NAME, COLLECTION_NAME, LLM_PORT, ENCODER_PORT, QDRANT_PORT


def setup_clients():
    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{ENCODER_PORT}/v1"

    llm_client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        timeout=10000
    )

    q_client = QdrantClient(
        host="localhost", 
        port=QDRANT_PORT, 
        check_compatibility=False, 
        timeout=1000
    )
    
    return llm_client, q_client


def load_data(csv_path: str):
    print(f"загружаем данные из {csv_path}")
    embs_table = pd.read_csv(csv_path, index_col=0).reset_index()
    retrieve_texts = embs_table["text"]
    return embs_table, retrieve_texts


def generate_embeddings(llm_client, retrieve_texts, batch_size: int = 2048):
    print("генерим эмбеддинги...")
    all_responses = []
    tests_batches = list(batched(retrieve_texts.to_list(), batch_size))
    
    for batch in tqdm(tests_batches, desc="Processing batches"):
        responses = llm_client.embeddings.create(
            input=batch,
            model=ENCODER_NAME,
        )
        all_responses.extend(responses.data)
    
    return all_responses


def prepare_data_for_qdrant(embs_table, all_responses):
    print("подготовка данных для Qdrant...")
    texts = embs_table["text"].to_list()
    embeddings = list(map(lambda x: x.embedding, all_responses))
    ids = embs_table.index.to_list()
    urls = embs_table.url.to_list()
    
    return texts, embeddings, ids, urls


def setup_qdrant_collection(q_client, collection_name: str, vector_size: int = 2048):    
    if q_client.collection_exists(collection_name):
        print(f"удаляем существующую коллекцию {collection_name}")
        q_client.delete_collection(collection_name)
    
    print(f"создаем коллекцию {collection_name}")
    q_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def upload_points_to_qdrant(q_client, collection_name: str, texts, embeddings, ids, urls, batch_size: int = 512):
    points = [
        PointStruct(id=idx, vector=vec, payload={"text": text, "urls": url})
        for idx, vec, text, url in zip(ids, embeddings, texts, urls)
    ]

    print(f"загружаем {len(points)} точек в базу...")
    batches = list(batched(points, batch_size))
    for batch in tqdm(batches, desc="Uploading batches"):
        q_client.upsert(collection_name=collection_name, points=list(batch), wait=True)


def create_text_index(q_client, collection_name: str):
    print("создагие текстового индекса...")
    q_client.create_payload_index(
        collection_name=collection_name,
        field_name="text",
        field_schema=TextIndexParams( 
            type="text",
            tokenizer=TokenizerType.MULTILINGUAL,
            lowercase=True,
            min_token_len=2,
            max_token_len=20,
        ),
        wait=True
    )

def main():
    parser = argparse.ArgumentParser(description="Заполнение Qdrant")
    parser.add_argument("--csv-path", default="data/buffer_table.csv")
    parser.add_argument("--collection-name", default="Docs_Dense")
    parser.add_argument("--embedding-batch-size", type=int, default=1024)
    parser.add_argument("--upload-batch-size", type=int, default=512)
    parser.add_argument("--vector-size", type=int, default=2048)
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"файл {csv_path} не найден!")
        sys.exit(1)
    
    try:
        llm_client, q_client = setup_clients()
        embs_table, retrieve_texts = load_data(str(csv_path))
        
        # эмбеддинги
        all_responses = generate_embeddings(llm_client, retrieve_texts, args.embedding_batch_size)
        texts, embeddings, ids, urls = prepare_data_for_qdrant(embs_table, all_responses)
        
        setup_qdrant_collection(q_client, args.collection_name, args.vector_size)
        upload_points_to_qdrant(
            q_client, args.collection_name, texts, embeddings, ids, urls, args.upload_batch_size
        )
        create_text_index(q_client, args.collection_name)
        
        collections = q_client.get_collections()
        print(f"\nГотово! Доступные коллекции: {collections}")
        
        collection_info = q_client.get_collection(args.collection_name)
        print(f"Коллекция {args.collection_name}: {collection_info.points_count} точек")
        
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()