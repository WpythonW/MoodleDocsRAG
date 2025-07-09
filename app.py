import asyncio
from typing import List, Dict, Optional
from qdrant_client import models, AsyncQdrantClient
from qdrant_client.http.models import (
    Prefetch, FusionQuery, Fusion,
    Filter, FieldCondition, MatchText
)
import httpx, json
from fastapi import FastAPI
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import SearchRequest
from config import LLM_NAME, ENCODER_NAME, COLLECTION_NAME, LLM_PORT, ENCODER_PORT, QDRANT_PORT

LLM = AsyncOpenAI(base_url=f"http://localhost:{LLM_PORT}/v1", api_key="EMPTY")
ENC = AsyncOpenAI(base_url=f"http://localhost:{ENCODER_PORT}/v1", api_key="EMPTY")
QDR = AsyncQdrantClient(host="localhost", port=QDRANT_PORT, check_compatibility=False, timeout=1000)

class RagRequest(BaseModel):
    query: str = Field(..., examples=["Как настроить кэш в Moodle?"])
    history: Optional[List[Dict[str, str]]] = None  # [{role:"user",content:"…"},{…}]
    k: int = 3
    top: int = 5

class QuestionFormat(BaseModel):
    queries: List[str]


async def gen_queries(q: str, k: int = 5) -> List[str]:
    sys_msg = (
        "You are a Moodle documentation RAG chatbot. "
        "Translate the query to English (if needed) and rephrase it using official Moodle documentation language. "
        f"Return exactly {k} short, precise, different versions as a JSON array of strings."
    )

    rsp = await LLM.chat.completions.create(
        model=LLM_NAME,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": q}
        ],
        max_tokens=512,
        temperature=0.5,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "people",
                "schema": QuestionFormat.model_json_schema()
            }
    },
    )

    content = json.loads(rsp.choices[0].message.content.strip())['queries']

    return content

async def embed(texts: List[str]) -> List[List[float]]:
    rsp = await ENC.embeddings.create(model=ENCODER_NAME, input=texts)
    return [d.embedding for d in rsp.data]


def build_system_prompt() -> str:
    return (
        "Answer IN RUSSIAN the question using ONLY the Moodle documentation below."
        "If the answer is not contained, say you don’t know. Use only info from context. DO NOT make up any information."
        "You have to use given context for answer!"
    )


async def query_request(questions_group, embeddings_group, top):
    prefetch = []
    for q_txt, q_vec in zip(questions_group, embeddings_group):
        prefetch.append(
            Prefetch(
                query=q_vec,
                using=None,
                limit=top
            )
        )

        prefetch.append(
            Prefetch(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="text",
                            match=MatchText(text=q_txt)
                        )
                    ]
                ),
                using="text",
                limit=top
            )
        )

    resp = await QDR.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch,
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top,
        with_payload=True
    )
    return resp

async def get_context(query, k, top):
    queries = await gen_queries(query, k)
    queries = queries + ['query']
    embeddings = await embed(queries)
    return queries, await query_request(queries, embeddings, top)


async def rag_pipeline(query: str,
                       history: Optional[List[Dict[str, str]]] = None,
                       k: int = 3, top: int = 5) -> Dict[str, str]:

    history = history or [] 
    queries, ctx_docs = await get_context(query, k, top)
    ctx_text = [i.payload['text'] for i in ctx_docs.points]

    system_prompt = build_system_prompt()
    messages = [{"role": "system", "content": system_prompt}] \
               + history \
               + [{"role": "user", "content": f"### Documentation\n{ctx_text}\n\nQuery: {query}"}]

    rsp = await LLM.chat.completions.create(
        model=LLM_NAME,
        messages=messages,
        temperature=0.1,
        max_tokens=512,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    answer = rsp.choices[0].message.content.strip()
    
    return {"answer": answer, "queries": queries, "context": ctx_docs}

app = FastAPI()

@app.post("/rag")
async def rag_endpoint(body: RagRequest):
    return await rag_pipeline(
        body.query,
        body.history,
        body.k,
        body.top
    )