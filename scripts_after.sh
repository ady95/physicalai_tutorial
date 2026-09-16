#!/usr/bin/env bash
# (집필 실측용) scripts_train_all.sh 가 끝난 뒤:
#   ACT v2 (117 episode, top 카메라, 30k) 학습·평가 → 성공률이 낮으면 ACT v3 (top_zoom 카메라, 30k) → 8부 파이프라인
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
export MUJOCO_GL=egl SVT_LOG=0 PYTHONUNBUFFERED=1
mkdir -p outputs/logs

train_eval() {  # name dataset_root repo_id top_camera
  name=$1; ds=$2; repo=$3; cam=$4
  rm -rf outputs/train/$name
  echo "=== $name train $(date +%H:%M:%S) ==="
  lerobot-train --dataset.repo_id=$repo --dataset.root=$ds \
    --policy.type=act --policy.chunk_size=50 --policy.n_action_steps=50 --policy.device=cuda --policy.push_to_hub=false \
    --output_dir=outputs/train/$name --job_name=$name --steps=30000 --batch_size=16 \
    --save_freq=10000 --log_freq=500 --wandb.enable=false > outputs/logs/ch06_train_$name.txt 2>&1
  best=0
  for ck in 010000 020000 030000; do
    echo "=== $name eval $ck $(date +%H:%M:%S) ==="
    python ch06/eval_act.py --checkpoint outputs/train/$name/checkpoints/$ck/pretrained_model --episodes 20 --top-camera $cam \
      --video --out outputs/ch06_${name}_$ck.mp4 2>&1 | grep -E "episode|성공률|chunk" > outputs/logs/ch06_eval_${name}_$ck.txt
    tail -1 outputs/logs/ch06_eval_${name}_$ck.txt
    r=$(grep -oE "= [0-9]+%" outputs/logs/ch06_eval_${name}_$ck.txt | grep -oE "[0-9]+")
    [ "${r:-0}" -gt "$best" ] && best=$r
  done
  echo "$name best=$best"
  echo $best > outputs/logs/${name}_best.txt
}

train_eval act_so101_v2 outputs/datasets/so101_pickplace_sim_150 physicalai/so101_pickplace_sim_150 top
if [ "$(cat outputs/logs/act_so101_v2_best.txt)" -lt 60 ]; then
  train_eval act_so101_v3 outputs/datasets/so101_pickplace_sim_150z physicalai/so101_pickplace_sim_150z top_zoom
fi
echo "=== ch08 $(date +%H:%M:%S) ==="
bash scripts_ch08.sh > outputs/logs/ch08_all.out 2>&1
echo "done $(date +%H:%M:%S)"
