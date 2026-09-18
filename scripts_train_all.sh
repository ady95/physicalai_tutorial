#!/usr/bin/env bash
# 6부·7부 학습 파이프라인 전체 (집필 실측용). 몇 시간 걸린다.
#   1) 시연 데이터셋 50 episode 기록
#   2) ACT 학습 → 평가
#   3) SmolVLA 사전학습 모델 그대로 평가 → 파인튜닝 → 평가
set -uo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export MUJOCO_GL=egl SVT_LOG=0
mkdir -p outputs/logs
F='Warning|warn|torchcodec|libtorchcodec|libav|it/s\]|B/s\]|Map:|mp4 @|EGLError|glCheckError|^\s*$'
run() {
  name=$1; shift
  echo "=== $name  $(date +%H:%M:%S) ==="
  "$@" > "outputs/logs/$name.raw" 2>&1
  rc=$?
  grep -vE "$F" "outputs/logs/$name.raw" > "outputs/logs/$name.txt" || true
  if [ $rc -ne 0 ]; then
    echo "!!! $name 실패 (exit $rc). 원본 로그 outputs/logs/$name.raw 의 마지막 40줄:"
    tail -40 "outputs/logs/$name.raw"
    exit $rc
  fi
  tail -${TAIL:-8} "outputs/logs/$name.txt"
}
DS=outputs/datasets/so101_pickplace_sim
ACT_STEPS=${ACT_STEPS:-20000}
VLA_STEPS=${VLA_STEPS:-10000}
RENAME='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'

run ch06_record        python ch06/record_demos.py --episodes 50 --overwrite --root $DS
run ch06_inspect_local python ch06/inspect_dataset.py --root $DS --repo-id physicalai/so101_pickplace_sim
run ch06_train_act     lerobot-train --dataset.repo_id=physicalai/so101_pickplace_sim --dataset.root=$DS \
    --policy.type=act --policy.chunk_size=50 --policy.n_action_steps=50 --policy.device=cuda --policy.push_to_hub=false \
    --output_dir=outputs/train/act_so101 --job_name=act_so101 --steps=$ACT_STEPS --batch_size=16 \
    --save_freq=5000 --log_freq=500 --wandb.enable=false
run ch06_eval_act      python ch06/eval_act.py --checkpoint outputs/train/act_so101/checkpoints/last/pretrained_model --episodes 20 --video
run ch07_eval_base     python ch07/eval_smolvla.py --checkpoint lerobot/smolvla_base --episodes 10 --video --out outputs/ch07_smolvla_base.mp4
run ch07_train_smolvla lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=physicalai/so101_pickplace_sim --dataset.root=$DS \
    --rename_map="$RENAME" --policy.device=cuda --policy.push_to_hub=false \
    --output_dir=outputs/train/smolvla_so101 --job_name=smolvla_so101 --steps=$VLA_STEPS --batch_size=8 \
    --save_freq=2500 --log_freq=250 --wandb.enable=false
run ch07_eval_smolvla  python ch07/eval_smolvla.py --checkpoint outputs/train/smolvla_so101/checkpoints/last/pretrained_model --episodes 20 --video
echo "done $(date +%H:%M:%S)"
