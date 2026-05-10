# Lab P2-09 - RAG avançado (HNSW + HyDE + Cross-Encoder)

Pipeline RAG para busca em manuais médicos. A query coloquial passa pelo HyDE (gpt-4o-mini gera um documento hipotético técnico), o vetor desse documento busca no índice HNSW (FAISS) com embeddings do BERTimbau, e o cross-encoder reordena os 10 candidatos do funil largo nos 3 finais.

## HNSW: M, ef_construction vs KNN exata

KNN exato guarda só os N embeddings (~N·d·4 bytes em float32). Sem overhead, mas busca custa O(N·d) por query.

HNSW guarda embeddings + um grafo: cada nó mantém até `M` vizinhos no nível base e ~M/2 nos superiores, somando RAM extra ≈ N·2M·8 bytes. Para 1M docs com d=768, M=16 adiciona ~256MB sobre ~3GB de embeddings (~8%); M=64 quase quadruplica. `ef_construction` só afeta tempo de build e qualidade do grafo — RAM final não muda porque o número máximo de vizinhos por nó continua sendo M.

Em troca de 5-30% mais RAM (proporcional a M), HNSW dá busca em O(log N) com recall >95%.

## Modelos escolhidos

Bi-encoder: `neuralmind/bert-base-portuguese-cased` (BERTimbau) — reaproveitado dos Labs 5/6. Mean pooling sobre o `last_hidden_state` mascarado + normalização L2 para usar produto interno como cosseno.

Cross-encoder: `BAAI/bge-reranker-v2-m3`. O `cross-encoder/ms-marco-MiniLM-L-6-v2` do PDF é treinado só em inglês e rerankeava errado em PT (todos os scores negativos = "irrelevante").

## Como rodar

    pip install torch transformers sentence-transformers faiss-cpu openai numpy
    export OPENAI_API_KEY=...
    python rag.py

`gera_manuais.py` regera os fragmentos via OpenAI — opcional, `manuais.jsonl` já vem comitado.

## Saída

As 4 queries coloquiais do `rag.py` (literal, sem acentos como no código):

| Query | Top-3 cobre? |
|---|---|
| `dor de cabeca latejante e luz incomodando` | sim (#2 migrânea) |
| `ardencia pra urinar e to indo no banheiro toda hora` | sim (#1 cistite) |
| `to com aperto no peito que vai pro braco esquerdo` | sim (#1 SCA) |
| `to me sentindo cansado o tempo todo, com frio e ganhei peso` | sim (#1 hipotireoidismo) |

Exemplo completo (cefaleia, HyDE real do gpt-4o-mini):

```
[hyde]
Na enxaqueca sem aura (ICHD-3 1.1), a cefaleia unilateral pulsátil
associada a fotofobia e fonofobia frequentemente responde a triptanos
ou AINEs.

[top-10 bi-encoder]
  cos=0.9138  Migranea caracteriza-se por cefaleia pulsatil...
  cos=0.8888  Cefaleia tensional apresenta-se como dor opressiva...
  cos=0.8761  Anemia ferropriva caracteriza-se por hemoglobina...
  ...

[top-3 cross-encoder]
  #1  score=0.0282  Cefaleia tensional apresenta-se como dor opressiva...
  #2  score=0.0229  Migranea caracteriza-se por cefaleia pulsatil...
  #3  score=0.0031  Sindrome coronariana aguda...
```

Nas queries 2-4 o cross-encoder promove o fragmento certo de posições baixas do top-10 (rank 3-5 do bi-encoder) para o #1 — é o filtro fino fazendo o trabalho.

## Uso de IA

Ferramenta usada: Claude Sonnet 4.6

- Elaboração do system prompt do HyDE
- Identificação de que o cross-encoder do PDF é treinado só em inglês e recomendação do BGE multilingual
- Estruturação do README (organização das seções, tabela de saída, formatação do bloco de exemplo)
