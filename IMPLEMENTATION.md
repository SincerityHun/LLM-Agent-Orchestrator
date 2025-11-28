# LLM-Agent-Orchestrator Implementation

## 시스템 개요

이 시스템은 Multi-Agent LLM 아키텍처로, 복잡한 작업을 역할별 Sub-Agent로 분해하여 실행합니다.

### 주요 특징

1. **DAG 기반 작업 분해**: Global Router가 작업을 분석하고 의존성 그래프(DAG) 생성
2. **LangGraph 오케스트레이션**: DAG를 LangGraph로 변환하여 병렬/직렬 실행
3. **도메인 특화 모델**: 각 작업에 적합한 전문 모델 자동 선택
4. **반복 개선**: Results Handler가 결과 평가 후 필요시 재실행

## 폴더 구조

```
./
├── main.py                      # 메인 진입점
├── requirements.txt             # Python 의존성
├── docker/
│   ├── Dockerfile              # 컨테이너 이미지
│   ├── docker-compose.yml      # vLLM 서버 + 오케스트레이터
│   └── .env                    # docker compose 환경변수 (HF_TOKEN 등)
├── utils/
│   ├── router_prompts.py       # Global Router 프롬프트
│   ├── agent_prompts.py        # Agent 프롬프트
│   ├── llm_loader.py           # vLLM 래퍼
│   └── merge_utils.py          # 결과 병합 유틸리티
├── routers/
│   ├── global_router.py        # Task 분해 및 DAG 생성
│   └── agent_subrouter.py      # 모델 선택 및 실행
├── agents/
│   ├── agent_planning.py       # Planning Agent
│   ├── agent_execution.py      # Execution Agent
│   └── agent_review.py         # Review Agent
├── orchestrator/
│   ├── graph_builder.py        # LangGraph 빌더
│   └── result_handler.py       # 결과 평가 및 피드백
├── examples/
│   └── sample_request.json     # 샘플 요청
├── test/
│   └── integration_test.py     # 통합 테스트
└── LLM_pool/                    # 도메인 특화 모델들
    ├── llama-1b-csqa/
    ├── llama-1b-casehold/
    ├── llama-1b-medqa2/
    └── qwen-3b-mathqa-2/
```

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. Docker Compose로 실행 (권장)

```bash
cd docker
docker-compose up -d
```

이 명령은 다음을 실행합니다:
- vLLM-Llama 서버 (Llama-3.2-1B + LoRA 어댑터 3개: CommonsenseQA, CaseHOLD, MedQA)
- vLLM-Qwen 서버 (Qwen-3B + MathQA LoRA 어댑터)
- Orchestrator 메인 애플리케이션

### 3. 직접 실행 (vLLM 서버가 이미 실행 중인 경우)

```bash
# 환경변수 설정
export VLLM_LLAMA_ENDPOINT=http://localhost:8000/v1
export VLLM_QWEN_ENDPOINT=http://localhost:8001/v1

# 실행
python main.py
```

### 4. JSON 파일로 실행

```bash
python main.py examples/sample_request.json
```

### 5. 커맨드라인으로 직접 실행

```bash
python main.py "Explain chest pain diagnosis for emergency patients"
```

## 테스트 실행

### 통합 테스트

```bash
python test/integration_test.py
```

### 개별 모듈 Unit Test

각 모듈은 독립적인 unit test를 포함합니다:

```bash
python utils/llm_loader.py
python utils/merge_utils.py
python routers/global_router.py
python routers/agent_subrouter.py
python agents/agent_planning.py
python orchestrator/graph_builder.py
python orchestrator/result_handler.py
```

## 시스템 플로우

1. **Global Router**: 작업 분해 및 DAG 생성
   - 입력 작업 분석
   - Planning/Execution/Review 역할별 subtask 생성
   - 의존성 그래프 생성

2. **Graph Builder**: LangGraph 빌드 및 실행
   - DAG를 LangGraph로 변환
   - 병렬/직렬 실행 계획 수립
   - 각 노드에서 적절한 Agent 호출

3. **Agent SubRouter**: 모델 선택 및 실행
   - 작업 내용 분석
   - 도메인 특화 모델 선택 (medical/legal/math/commonsense)
   - vLLM 통해 추론 실행

4. **Merge & Evaluate**: 결과 통합 및 평가
   - 모든 Agent 결과 병합
   - Results Handler가 완성도 평가
   - 필요시 피드백과 함께 재실행

5. **Loop Control**: 반복 제어
   - MAX_RETRY(기본 3회) 이내 반복
   - 충분한 결과 획득 시 종료
   - 최대 횟수 도달 시 현재 결과 반환

## 핵심 컴포넌트 설명

### Global Router (routers/global_router.py)
- LLM을 사용해 작업을 분석하고 DAG 생성
- 피드백 기반 재분해 지원
- JSON 형식 DAG 출력

### Graph Builder (orchestrator/graph_builder.py)
- DAG를 LangGraph StateGraph로 변환
- 진입점/종료점 자동 탐지
- Agent 실행 결과를 State로 관리

### Agent SubRouter (routers/agent_subrouter.py)
- 키워드 기반 도메인 탐지
- 적절한 vLLM 엔드포인트 선택
- Context 전달 및 결과 수집

### Result Handler (orchestrator/result_handler.py)
- 병합된 결과 평가
- 완성도 판단 및 피드백 생성
- MAX_RETRY 제어

## 모델 엔드포인트

- **Llama 서비스** (Llama-3.2-1B + LoRA): http://localhost:8000
  - CommonsenseQA: `csqa-lora` 모델명 사용
  - CaseHOLD: `casehold-lora` 모델명 사용  
  - MedQA-USMLE: `medqa-lora` 모델명 사용
- **Qwen 서비스** (Qwen-3B + LoRA): http://localhost:8001
  - MathQA: `mathqa-lora` 모델명 사용

## 환경 변수

- `VLLM_LLAMA_ENDPOINT`: Llama 통합 서비스 엔드포인트 (CommonsenseQA, CaseHOLD, MedQA)
- `VLLM_QWEN_ENDPOINT`: Qwen 서비스 엔드포인트 (MathQA)
- `HF_TOKEN`: Hugging Face 액세스 토큰 (선택사항)

## 코딩 규칙 준수 사항

- 주석: 핵심 로직에만 작성, 이모티콘 미사용
- Unit Test: 각 파일의 `if __name__ == "__main__":` 블록에서만 수행
- Integration Test: `test/integration_test.py`에만 작성
- vLLM 호출: `utils/llm_loader.py`의 VLLMLoader 사용
- LangGraph: 오케스트레이션 프레임워크로 사용

## 트러블슈팅

### vLLM 서버가 시작되지 않는 경우
- GPU 메모리 확인 (각 서비스별 80% 메모리 사용)
- CUDA 드라이버 버전 확인
- docker-compose.yml의 gpu-memory-utilization 조정 (현재 0.8)
- LoRA 어댑터 경로 확인 (`/csqa-adapter`, `/casehold-adapter`, `/medqa-adapter`, `/adapter`)
- GPU 디바이스 할당 확인 (Llama: GPU 2, Qwen: GPU 3)

### 모델 선택이 올바르지 않은 경우
- `routers/agent_subrouter.py`의 domain_keywords 수정
- 작업 설명에 명확한 도메인 키워드 포함
- LoRA 어댑터 모델명 확인:
  - CommonsenseQA: `csqa-lora`
  - CaseHOLD: `casehold-lora`
  - MedQA: `medqa-lora`
  - MathQA: `mathqa-lora`

### 무한 루프 방지
- Result Handler의 MAX_RETRY 설정 확인
- 기본값: 3회
