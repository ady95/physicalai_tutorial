#!/usr/bin/env bash
# (집필 실측용) ACT v3 가 끝난 뒤: 3색 블록 데이터셋 240 episode 로 SmolVLA 재파인튜닝(20k) → 색별 평가 → 언어 진단
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
export MUJOCO_GL=egl SVT_LOG=0 PYTHONUNBUFFERED=1
while pgrep -f "scripts_after3.sh" >/dev/null; do sleep 60; done
while pgrep -f "record_multicolor.py" >/dev/null; do sleep 30; done
DSM=outputs/datasets/so101_multicolor_sim_240
RENAME='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
rm -rf outputs/train/smolvla_multicolor_v2
echo "=== smolvla_multicolor_v2 train $(date +%H:%M:%S) ==="
lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=physicalai/so101_multicolor_sim_240 --dataset.root=$DSM \
  --rename_map="$RENAME" --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=outputs/train/smolvla_multicolor_v2 --job_name=smolvla_multicolor_v2 --steps=20000 --batch_size=8 \
  --save_freq=5000 --log_freq=250 --wandb.enable=false > outputs/logs/ch08_train_multi_v2.txt 2>&1
for ck in 010000 020000; do
  echo "=== multicolor_v2 eval $ck $(date +%H:%M:%S) ==="
  python ch08/eval_multicolor.py --checkpoint outputs/train/smolvla_multicolor_v2/checkpoints/$ck/pretrained_model --episodes-per-color 10 --video --out outputs/ch08_multicolor_v2_$ck.mp4 2>&1 | grep -E "layout|색|red|green|yellow|전체" > outputs/logs/ch08_eval_multi_v2_$ck.txt
  tail -1 outputs/logs/ch08_eval_multi_v2_$ck.txt
  python ch08/diagnose_language.py --checkpoint outputs/train/smolvla_multicolor_v2/checkpoints/$ck/pretrained_model 2>&1 | grep -E "layout|비율" > outputs/logs/ch08_diag_multi_v2_$ck.txt
  tail -1 outputs/logs/ch08_diag_multi_v2_$ck.txt
done
echo "done $(date +%H:%M:%S)"
