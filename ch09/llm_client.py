"""9부 — LLM(Responses API)을 부르는 얇은 래퍼.

이 책의 다른 부와 달리 9부는 외부 LLM API를 씁니다. 접속 정보는 코드에 적지 않고
OpenAI SDK가 표준으로 읽는 환경변수에서 가져옵니다.

    OPENAI_API_KEY    발급받은 API 키
    OPENAI_BASE_URL   (선택) 다른 서버를 쓸 때만. 비워 두면 공식 OpenAI 서버
    OPENAI_MODEL      (선택) 모델 이름. 기본값은 아래 DEFAULT_MODEL

왜 Chat Completions가 아니라 Responses API인가:
카메라 이미지를 base64(data: URL)로 넣어야 하는데, 이 방식은 Responses API의
input_image에서만 확실히 동작합니다. 9-4에서 실제로 확인합니다.
"""

import json
import os
import time

DEFAULT_MODEL = "gpt-6-astra"   # 9부의 수치를 측정한 모델
MAX_RETRY = 3


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Responses API 한 번 호출 = respond() 한 번. 대화 이력은 호출하는 쪽이 들고 있습니다."""

    def __init__(self, model=None, timeout=120.0):
        try:
            from openai import OpenAI
        except ImportError as e:  # 9부에서만 쓰므로 openai 를 기본 설치에 넣지 않았습니다
            raise LLMError("openai 패키지가 없습니다.  uv pip install openai") from e
        if not os.getenv("OPENAI_API_KEY"):
            raise LLMError("환경변수 OPENAI_API_KEY 가 없습니다. .env 를 확인하세요.")
        self.client = OpenAI(timeout=timeout)       # base_url/api_key 는 환경변수에서 자동
        self.model = model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0}

    def respond(self, input_items, tools):
        """대화 이력 전체를 보내고 output 항목 리스트(dict)를 돌려준다."""
        last = None
        for attempt in range(MAX_RETRY):
            t0 = time.time()
            try:
                r = self.client.responses.create(model=self.model, input=input_items, tools=tools)
            except Exception as e:                   # 네트워크·일시적 5xx 는 몇 초 쉬고 재시도
                last = e
                time.sleep(2 * (attempt + 1))
                continue
            dt = time.time() - t0
            self.usage["requests"] += 1
            self.usage["seconds"] += dt
            u = getattr(r, "usage", None)
            if u is not None:
                self.usage["input_tokens"] += getattr(u, "input_tokens", 0) or 0
                self.usage["output_tokens"] += getattr(u, "output_tokens", 0) or 0
            return [o.model_dump(exclude_none=True) for o in r.output], dt
        raise LLMError(f"LLM 호출이 {MAX_RETRY}회 실패했습니다: {last}")


def clean_transcript(items):
    """trace 를 저장하기 전에 정리한다.

    - base64 이미지를 자리표시자로 (파일이 수십 MB가 되는 것을 방지)
    - 모델이 돌려주는 암호화 추론 블록(encrypted_content)을 제거.
      우리가 읽을 수 없는 불투명 데이터인데 용량만 크고, 비밀 스캐너가 오탐한다.
    """
    out = []
    for it in items:
        it = json.loads(json.dumps(it))
        it.pop("encrypted_content", None)
        content = it.get("content")
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and isinstance(c.get("image_url"), str) and c["image_url"].startswith("data:"):
                    c["image_url"] = f"<image {len(c['image_url'])} chars>"
        out.append(it)
    return out


def save_trace(path, record):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)
    print(f"trace 저장: {path}")
