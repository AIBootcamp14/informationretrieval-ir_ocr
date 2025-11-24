# Title Scientific Knowledge Question Answering(OCR)

## Team

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [박패캠](https://github.com/UpstageAILab)             |            [이패캠](https://github.com/UpstageAILab)             |            [최패캠](https://github.com/UpstageAILab)             |            [김패캠](https://github.com/UpstageAILab)             |            [오패캠](https://github.com/UpstageAILab)             |
|                            팀장, 담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |

## 0. Overview
### Environment

- OS: Linux/macOS

- Language: Python 3.10+

- Database: Elasticsearch 8.8.0

- LLM: OpenAI GPT-4o-mini (Classification, Reranking, Generation)

### Requirements
- RAG 시스템의 MAP 점수를 극대화하는 것

## 1. Competiton Info

### Overview

- 본 대회는 RAG(Retrieval Augmented Generation) 시스템의 검색 성능에 집중하며, 4200여 개의 과학 상식 문서를 대상으로 질의 응답 시스템을 구축하는 것이 목표 입니다.

- 특히, 질문이 과학 상식인지 아닌지를 판단하는 의도 분류가 MAP 점수에 결정적인 영향을 끼칩니다.


대회에서 사용되는 변형된 MAP 로직은 두 가지 핵심 지표로 구성
저희 팀은 이 로직을 역분석하여 성능 극대화 전략을 세웠습니다.

- 1. 과학 상식 질문인 경우 (MAP의 Precision 측정):

topk (최대 3개)에 정답 문서가 얼마나 정확하게 포함되었는지로 점수 계산

목표: Reranking을 통해 검색 결과의 **정확도(Precision)**를 극대화해야 함

- 2. 과학 상식 질문이 아닌 경우 (의도 분류 정확도):

검색 결과 (topk)가 없어야 (False) $\text{1}$점을 획득. topk가 존재하면 $\text{0}$점

목표: query_analyzer.py 모듈을 통해 비과학 질문을 정확히 False로 분류해야 함

### Timeline

- 2025.11.14 10:00
- 2025.11.27 19:00

## 2. Components

### Directory

- .env 기반의 모듈화된 아키텍처를 구축하여 안정성과 유지보수성을 확보

e.g.
```
/ir_ocr_rag_project
├── .gitignore
├── .env                     # 환경 변수 (최고 보안 및 유연성)
├── requirements.txt         # 패키지 목록
├── run_competition.py       # 메인 실행 스크립트
├── config.py                # 설정 관리 및 환경 변수 로드
│
├── src                      # 핵심 로직 모듈
│   ├── __init__.py          
│   ├── query_analyzer.py    # 의도 분류 (MAP 0/1 결정) 및 Standalone Query 생성
│   ├── retriever.py         # OCR 앙상블 검색 및 LLM Reranking (TopK 결정)
│   └── generator.py         # 최종 답변 생성
│
├── data                     # 데이터 폴더 (Git 무시)
│   ├── .gitkeep
│   ├── documents.jsonl
│   └── eval.jsonl
│
└── scripts                  # 유틸리티
    ├── install_elasticsearch.sh
    └── index_data.py        # 데이터 색인 전용
```

## 3. Data descrption

### Dataset overview

- 약 4200개의 과학 상식 문서 (MMLU, ARC 기반). 문서 전문 텍스트는 content 필드에 저장됨


- 평가 데이터 (eval.jsonl): 총 $\text{220}$개의 질의.

-  특징 1: 멀티턴(Multi-turn) 대화 포함 (맥락 이해 필수).

-  특징 2: 비과학 질문(잡담, 의견 등) 약 $\text{20}$개 포함 (MAP 점수 관리가 핵심)

### EDA

- _Describe your EDA process and step-by-step conclusion_

### Data Processing

- 색인 전략: Elasticsearch를 사용하여 documents.jsonl의 문서들을 scientific_knowledge_index에 색인

- docid, src, content. content 필드는 BM25 키워드 검색에 최적화된 표준 분석기(standard analyzer)를 사용 (임베딩 대신 키워드 검색의 강점을 활용)


## 4. Modeling

### Model descrition


- 속도와 정확도를 동시에 잡기 위해 $\text{GPT-4o-mini}$ 모델을 파이프라인의 핵심적인 판단 및 재조정(Reranking) 단계에 활용

- 의도 분류: 모델 역할은 사용자의 질문이 과학 상식인지 아닌지를 판단하는 거야. MAP 평가 로직상 비과학 질문에 검색 결과를 주면 0점이 되기 때문에, 이 단계는 MAP 1점 방어를 위한 가장 중요한 방어선. GPT-4o-mini가 빠르고 저렴하면서도 '아니다(False)' 판단에 대한 정확도가 매우 높다는 점을 확인하고 이 모델을 선택

- Reranking (재조정): 검색 엔진(Elasticsearch)에서 얻은 15개의 초기 문서 풀에서 가장 정확한 3개의 문서를 최종적으로 선정하는 역할을 함. 전통적인 BM25 점수 대신 GPT-4o-mini의 높은 문맥 이해력을 사용하여 질문과 문서 내용의 실제 의미적 연관성을 평가. 이를 통해 Precision을 극대화하고 topk의 정확도를 비약적으로 높일 수 있었기 때문에 이 모델을 채택

- 답변 생성: 최종적으로 선택된 3개 문서를 기반으로 사용자에게 전달할 최종 답변을 생성하는 역할이야. GPT-4o-mini는 문서 기반의 사실을 확인하고, "초등학생도 이해하기 쉽게"라는 프롬프트 지침에 맞춰 간결하고 명확한 답변을 만드는 데 최적의 성능을 보여줬기 때문에 최종 모델로 선정



### Modeling Process

- 저희 팀은 단순히 검색을 한 번 하는 것이 아니라, MAP 지표의 변동성을 관리하고 OCR의 불확실성을 수용한 3단계 파이프라인을 구축했습니다.

Step 1: 의도 분류 및 Standalone Query 생성 (query_analyzer.py)

목표: MAP $\text{1}$점 방어

전략: LLM에게 "감정, 의견, 가치 판단은 무조건 False"라는 가장 엄격한 시스템 프롬프트를 주입.멀티턴 질문에 대해서는 완전한 맥락을 갖춘 Standalone Query를 생성하여 검색 엔진에 전달

Step 2: OCR 앙상블 검색 & Reranking (retriever.py)

OCR 극복 전략 (Recall 확보):

**원본 쿼리 (Standalone Query)**로 Elasticsearch 검색 (K=15).

OCR 노이즈 쿼리 생성: LLM(gpt-4o-mini, $T=0.8$)을 사용해 띄어쓰기 오류, 유사 글자 치환 등의 OCR 오류를 시뮬레이션한 변형 쿼리 생성.

앙상블 검색: 원본 쿼리와 OCR 노이즈 쿼리 두 개로 검색을 수행하고, 결과를 합쳐 중복 제거 후 상위 15개의 검색 풀(Pool)을 형성 (Recall 극대화)

정확도 극대화 (Precision 확보):
4.  LLM Reranking: 검색 풀 15개와 원본 질문을 다시 gpt-4o-mini에게 전달하여 내용적 연관성이 가장 높은 최종 3개topk의 docid만 선정

Step 3: 답변 생성 (generator.py)

효율성: topk로 선정된 docid를 Elasticsearch의 MGET API를 통해 한 번에 가져와서 네트워크 효율을 높임

간결성: LLM에게 "초등학생도 알기 쉽게, 짧고 간결하게" 답하라는 지침을 프롬프트에 명시하여 사용자 요구사항을 충족


## 5. Result

### Leader Board

- _Insert Leader Board Capture_
- _Write rank and score_


