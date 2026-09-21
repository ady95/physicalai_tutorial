#!/usr/bin/env bash
# 9부(선택 과정) 실측 파이프라인 — LLM Planner
#   9-4: 3-6과 같은 블록 위치 10곳에서 Pick and Place. perception 세 가지 비교
#   9-6: 8-1과 같은 배치·문장·판정으로 색 지정 과제. SmolVLA 와 비교
#
# 본편과 달리 학습이 없습니다. GPU 도 LeRobot 도 쓰지 않습니다.
# 대신 외부 LLM API 를 호출하므로 요금이 발생합니다. 아래 전체를 돌리면
# 약 190 Episode, 200만 토큰 규모입니다. 먼저 SMOKE=1 로 규모를 줄여 확인하세요.
#
#   SMOKE=1 ./scripts_ch09.sh      # 각 조건 2~3 회만 (수천 원 수준)
#   ./scripts_ch09.sh              # 원고에 실은 전체 측정
set -uo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export MUJOCO_GL=egl

: "${OPENAI_API_KEY:?환경변수 OPENAI_API_KEY 를 설정하세요}"
export OPENAI_MODEL=${OPENAI_MODEL:-gpt-5.1}
mkdir -p outputs/logs outputs/traces

if [ -n "${SMOKE:-}" ]; then TRIALS=2; LAYOUTS=1; else TRIALS=10; LAYOUTS=20; fi

run() {
  name=$1; shift
  echo "=== $name  $(date +%H:%M:%S) ==="
  "$@" > "outputs/logs/$name.txt" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "!!! $name 실패 (exit $rc). 로그 마지막 40줄:"
    tail -40 "outputs/logs/$name.txt"
    exit $rc
  fi
  tail -${TAIL:-6} "outputs/logs/$name.txt"
}

echo "모델 $OPENAI_MODEL  / 9-4 각 $TRIALS 회, 9-6 배치 $LAYOUTS 개"

# 9-4 — 눈을 바꿔 가며 같은 과제
for P in state color camera; do
  run "ch09_pick_$P" python ch09/run_llm_pick.py --no-video --trials $TRIALS --perception $P \
      --trace outputs/traces/ch09_pick_$P.json
done

# 9-6 — 8-1 과 같은 색 지정 과제 (camera 는 비용이 커서 배치를 절반으로)
run ch09_multicolor_state  python ch09/eval_llm_multicolor.py --layouts $LAYOUTS --perception state \
    --trace outputs/traces/ch09_multicolor_state.json
run ch09_multicolor_camera python ch09/eval_llm_multicolor.py --layouts $((LAYOUTS / 2)) --perception camera \
    --trace outputs/traces/ch09_multicolor_camera.json

echo "=== 완료  $(date +%H:%M:%S) ==="
echo "성공률 요약:"
grep -hE "성공률|언어 일치" outputs/logs/ch09_*.txt
