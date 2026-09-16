#!/usr/bin/env bash
# (집필 실측용) scripts_after2.sh(SmolVLA v2 + 8부)가 끝난 뒤: 확대 카메라(top_zoom) 데이터셋으로 ACT v3 학습·평가
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
export MUJOCO_GL=egl SVT_LOG=0 PYTHONUNBUFFERED=1
while pgrep -f "scripts_after2.sh" >/dev/null; do sleep 60; done
DS=outputs/datasets/so101_pickplace_sim_150z
rm -rf outputs/train/act_so101_v3
echo "=== act_so101_v3 train $(date +%H:%M:%S) ==="
lerobot-train --dataset.repo_id=physicalai/so101_pickplace_sim_150z --dataset.root=$DS \
  --policy.type=act --policy.chunk_size=50 --policy.n_action_steps=50 --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=outputs/train/act_so101_v3 --job_name=act_so101_v3 --steps=20000 --batch_size=16 \
  --save_freq=10000 --log_freq=500 --wandb.enable=false > outputs/logs/ch06_train_act_v3.txt 2>&1
for ck in 010000 020000; do
  echo "=== act_so101_v3 eval $ck $(date +%H:%M:%S) ==="
  python ch06/eval_act.py --checkpoint outputs/train/act_so101_v3/checkpoints/$ck/pretrained_model --episodes 20 --top-camera top_zoom \
    --video --out outputs/ch06_act_so101_v3_$ck.mp4 2>&1 | grep -E "episode|성공률|chunk" > outputs/logs/ch06_eval_act_so101_v3_$ck.txt
  tail -1 outputs/logs/ch06_eval_act_so101_v3_$ck.txt
done
echo "done $(date +%H:%M:%S)"
