# Pipeline RAG: HNSW (FAISS) + HyDE + Cross-Encoder

import json
import os
import numpy as np
import faiss
import openai
from sentence_transformers import SentenceTransformer, CrossEncoder

MODELO_EMB = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODELO_RERANK = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MODELO_LLM = "gpt-4o-mini"

M = 16
EF_CONSTRUCTION = 200
EF_SEARCH = 50
TOP_K = 10
TOP_FINAL = 3

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

modelo = SentenceTransformer(MODELO_EMB)
ce = CrossEncoder(MODELO_RERANK)

with open("manuais.jsonl", encoding="utf-8") as f:
    docs = [json.loads(l) for l in f]
textos = [d["texto"] for d in docs]

emb = modelo.encode(textos, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
dim = emb.shape[1]

idx = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
idx.hnsw.efConstruction = EF_CONSTRUCTION
idx.hnsw.efSearch = EF_SEARCH
idx.add(emb)


def hyde(query):
    resp = client.chat.completions.create(
        model=MODELO_LLM,
        messages=[
            {"role": "system", "content": "Voce eh um medico escrevendo um trecho curto de manual tecnico. Em 2 a 3 frases, descreva a condicao usando jargao medico (siglas, classes de medicamento, criterios). Nao se preocupe em ser exato — escreva um texto plausivel pra ancorar uma busca semantica."},
            {"role": "user", "content": query},
        ],
        temperature=0.7,
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()


def busca(query):
    hyde_doc = hyde(query)
    print(f"\n[hyde]\n{hyde_doc}\n")

    v = modelo.encode([hyde_doc], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, ids = idx.search(v, TOP_K)

    print(f"[top-{TOP_K} bi-encoder]")
    cands = []
    for s, i in zip(scores[0], ids[0]):
        cands.append(textos[i])
        print(f"  cos={s:.4f}  {textos[i][:90]}...")

    pares = [[hyde_doc, c] for c in cands]
    score_ce = ce.predict(pares)

    ordem = np.argsort(score_ce)[::-1]
    print(f"\n[top-{TOP_FINAL} cross-encoder]")
    for r, j in enumerate(ordem[:TOP_FINAL]):
        print(f"  #{r+1}  score={score_ce[j]:.4f}")
        print(f"        {cands[j]}\n")


if __name__ == "__main__":
    queries = [
        "dor de cabeca latejante e luz incomodando",
        "ardencia pra urinar e to indo no banheiro toda hora",
        "to com aperto no peito que vai pro braco esquerdo",
        "to me sentindo cansado o tempo todo, com frio e ganhei peso",
    ]

    for q in queries:
        print("=" * 70)
        print(f"query: {q}")
        busca(q)
