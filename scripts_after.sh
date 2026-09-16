#!/usr/bin/env bash
# (집필 실측용) scripts_train_all.sh 가 끝난 뒤: ACT v2(117 episode, 30k) 학습·평가 → 8부 파이프라인
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
export MUJOCO_GL=egl SVT_LOG=0 PYTHONUNBUFFERED=1
mkdir -p outputs/logs
while pgrep -f scripts_train_all.sh >/dev/null; do sleep 60; done
DS=outputs/datasets/so101_pickplace_sim_150
rm -rf outputs/train/act_so101_v2
echo "=== act_v2 train $(date +%H:%M:%S) ==="
lerobot-train --dataset.repo_id=physicalai/so101_pickplace_sim_150 --dataset.root=$DS \
  --policy.type=act --policy.chunk_size=50 --policy.n_action_steps=50 --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=outputs/train/act_so101_v2 --job_name=act_so101_v2 --steps=30000 --batch_size=16 \
  --save_freq=10000 --log_freq=500 --wandb.enable=false > outputs/logs/ch06_train_act_v2.txt 2>&1
for ck in 010000 020000 030000; do
  echo "=== act_v2 eval $ck $(date +%H:%M:%S) ==="
  python ch06/eval_act.py --checkpoint outputs/train/act_so101_v2/checkpoints/$ck/pretrained_model --episodes 20 --video --out outputs/ch06_act_v2_$ck.mp4 2>&1 | grep -E "episode|성공률|chunk" > outputs/logs/ch06_eval_act_v2_$ck.txt
  tail -1 outputs/logs/ch06_eval_act_v2_$ck.txt
done
echo "=== ch08 $(date +%H:%M:%S) ==="
bash scripts_ch08.sh > outputs/logs/ch08_all.out 2>&1
echo "done $(date +%H:%M:%S)"
