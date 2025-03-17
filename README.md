# RAG 시스템 구현 가이드

[NVIDIA RTX AI PC Campus Seminar with 서울대학교]에서 발표한 "NIM으로 빠르고 똑똑한 RAG 만들기"의 실습 자료입니다. 이 저장소에서는 검색 증강 생성(Retrieval-Augmented Generation, RAG) 시스템의 구현 방법과 다양한 고급 검색 기법을 소개합니다.

![image](https://github.com/user-attachments/assets/75ef001b-f331-4062-bb14-6e525730908a)


## 목차
- [소개](#소개)
- [설치](#설치)
- [RAG 개요](#rag-개요)
- [고급 검색 기법](#고급-검색-기법)
  - [Multi-Query Retriever](#multi-query-retriever)
  - [Reorder](#reorder)
  - [Reranker](#reranker)
  - [Ensemble Retriever (Hybrid Search)](#ensemble-retriever-hybrid-search)
  - [Metadata Filtering](#metadata-filtering)
- [학습 자료](#학습-자료)
- [마무리](#마무리)

## 소개

RAG(Retrieval-Augmented Generation)는 LLM(Large Language Model)의 응답 품질을 향상시키기 위해 외부 지식을 활용하는 방법입니다. 이 저장소에서는 다양한 RAG 구현 방법과 고급 검색 기법을 소개하고, 실제 예제를 통해 그 효과를 확인할 수 있습니다.

## 설치

필요한 패키지를 설치하려면 다음 명령어를 실행하세요:

```bash
pip install langchain-nvidia-ai-endpoints langchain langchain-community langchain-core unstructured sentence_transformers faiss-cpu openai selenium pypdf pacmap plotly_express nbformat rank_bm25 kiwipiepy
```

## RAG 개요

### RAG 정의
RAG는 LLM에게 관련 정보를 제공하여 더 정확하고 사실에 기반한 응답을 생성하도록 돕는 기술입니다. 파인 튜닝보다 적은 비용으로 LLM의 성능을 향상시킬 수 있으며, 지식의 격차, 사실적 오류, 환각(hallucination) 문제를 줄일 수 있습니다.

### RAG 구성 요소
1. **입력(Input)**: LLM이 답변해야 할 질문
2. **인덱싱(Indexing)**: 문서를 청크로 나누고 벡터로 변환하여 저장
3. **검색(Retrieval)**: 질문과 관련된 문서 청크를 검색
4. **생성(Generation)**: 검색된 문서와 질문을 결합하여 최종 답변 생성

## 고급 검색 기법

### Multi-Query Retriever
LLM을 활용하여 하나의 쿼리에서 여러 관점의 쿼리를 자동으로 생성하고, 각 쿼리마다 관련 문서를 검색한 후 결과를 통합하는 방식입니다. 이를 통해 다양한 관점을 반영하고 검색 결과의 품질을 높일 수 있습니다.

```python
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 2}),
    llm=inference_model,
    prompt=MULTI_QUERY_PROMPT
)
```

### Reorder
"Lost in the middle" 문제를 해결하기 위한 기법으로, 검색된 문서를 재정렬하여 가장 관련성 높은 문서를 처음과 끝에 배치하고 덜 관련된 문서를 중간에 배치합니다.

```python
reordering = LongContextReorder()
reordered_docs = reordering.transform_documents(docs)
```

### Reranker
검색된 문서의 순위를 더 강력한 모델을 사용하여 다시 매기는 방법입니다. 쿼리와 문서 간의 맥락을 더 깊이 고려하여 관련성 점수를 계산합니다.

```python
reranker = NVIDIARerank(
    model=NIM_RERANKING_MODEL,
    api_key=NIM_API_KEY
)

reranked_docs = reranker.compress_documents(documents=docs, query=query)
```

### Ensemble Retriever (Hybrid Search)
여러 검색기의 결과를 RRF(Reciprocal Rank Fusion) 알고리즘을 통해 통합하는 방식입니다. 키워드 기반 검색(BM25)과 의미 기반 검색(벡터 검색)의 장점을 결합하여 더 나은 검색 결과를 제공합니다.

```python
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)
```

### Metadata Filtering
문서의 메타데이터를 활용하여 검색 결과를 필터링하는 방법입니다. 별점, 리뷰 수, 영업 상태 등 다양한 조건으로 검색 결과를 정제할 수 있습니다.

## 학습 자료

### 기초 자료
1. [RAG 101: 검색 증강 생성 파이프라인의 이해](https://developer.nvidia.com/ko-kr/blog/rag-101-demystifying-retrieval-augmented-generation-pipelines/)
2. [RAG 101: 검색 증강 생성 관련 질문과 답변](https://developer.nvidia.com/ko-kr/blog/rag-101-retrieval-augmented-generation-questions-answered/)

### 심화 자료
1. [Agentic RAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/)
2. [Adaptive RAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/)
3. [SQL Agent](https://langchain-ai.github.io/langgraph/tutorials/sql-agent/)

## 마무리

### 데이터 품질의 중요성
RAG 시스템에서는 입력 데이터의 질이 결과의 질을 결정합니다. 다양한 검색 기법도 중요하지만, 데이터의 품질을 높이는 것이 우선입니다.

### 성능과 비용의 균형
고급 검색 기법은 성능을 향상시키지만 추가 비용과 지연 시간을 발생시킵니다. 각 기능을 추가할 때는 비용 대비 효과를 신중하게 고려해야 합니다.

## 연락처
- 이름: 김하림
- 이메일: harheem@gsneotek.com
- LinkedIn: https://www.linkedin.com/in/harheemk/
