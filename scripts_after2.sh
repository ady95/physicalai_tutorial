#!/usr/bin/env bash
# (집필 실측용) scripts_after.sh 가 끝난 뒤: SmolVLA v2 (117 episode 데이터셋) 파인튜닝·평가 → 8부 파이프라인
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
export MUJOCO_GL=egl SVT_LOG=0 PYTHONUNBUFFERED=1
mkdir -p outputs/logs
while pgrep -f "scripts_after.sh" >/dev/null; do sleep 60; done
while pgrep -f "record_multicolor.py" >/dev/null; do sleep 30; done
DS=outputs/datasets/so101_pickplace_sim_150
RENAME='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
rm -rf outputs/train/smolvla_so101_v2
echo "=== smolvla_v2 train $(date +%H:%M:%S) ==="
lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=physicalai/so101_pickplace_sim_150 --dataset.root=$DS   --rename_map="$RENAME" --policy.device=cuda --policy.push_to_hub=false   --output_dir=outputs/train/smolvla_so101_v2 --job_name=smolvla_so101_v2 --steps=10000 --batch_size=8   --save_freq=2500 --log_freq=250 --wandb.enable=false > outputs/logs/ch07_train_smolvla_v2.txt 2>&1
echo "=== smolvla_v2 eval $(date +%H:%M:%S) ==="
python ch07/eval_smolvla.py --checkpoint outputs/train/smolvla_so101_v2/checkpoints/last/pretrained_model --episodes 20 --video --out outputs/ch07_smolvla_v2.mp4 2>&1 | grep -E "policy|chunk|episode|성공률" > outputs/logs/ch07_eval_smolvla_v2.txt
tail -1 outputs/logs/ch07_eval_smolvla_v2.txt
rm -f outputs/HOLD_CH08
echo "=== ch08 $(date +%H:%M:%S) ==="
bash scripts_ch08.sh > outputs/logs/ch08_all.out 2>&1
echo "done $(date +%H:%M:%S)"
