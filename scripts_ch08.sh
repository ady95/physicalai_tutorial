#!/usr/bin/env bash
# 8부 실측 파이프라인 (scripts_train_all.sh 가 끝난 뒤 실행)
#   8-1: 세 가지 색 블록 데이터셋 60 episode → SmolVLA 파인튜닝 → 색별 평가
#   8-2: 고정 위치 20 episode → ACT 학습 → 고정/무작위 위치 평가, 6부 ACT 와 비교
#   8-3: 조명·바닥·카메라 변형에서 ACT 평가
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
export MUJOCO_GL=egl SVT_LOG=0
mkdir -p outputs/logs
F='Warning|warn|torchcodec|libtorchcodec|libav|it/s\]|B/s\]|Exception ignored|Traceback|File "|EGLError|glCheckError|^\s*$'
run() { name=$1; shift; echo "=== $name  $(date +%H:%M:%S) ==="; "$@" 2>&1 | grep -vE "$F" | tee "outputs/logs/$name.txt" | tail -${TAIL:-8}; }
RENAME='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
ACT=outputs/train/act_so101/checkpoints/last/pretrained_model
VLA_STEPS=${VLA_STEPS:-10000}

# 8-3 (빠름, 6부 ACT 만 필요)
run ch08_robustness    python ch08/robustness.py --checkpoint $ACT --episodes 10

# 8-2
DSF=outputs/datasets/so101_fixed_sim
run ch08_record_fixed  python ch06/record_demos.py --episodes 20 --root $DSF --repo-id physicalai/so101_fixed_sim --cube 0.24 0.0
run ch08_train_fixed   lerobot-train --dataset.repo_id=physicalai/so101_fixed_sim --dataset.root=$DSF \
    --policy.type=act --policy.chunk_size=50 --policy.n_action_steps=50 --policy.device=cuda --policy.push_to_hub=false \
    --output_dir=outputs/train/act_fixed --job_name=act_fixed --steps=8000 --batch_size=16 --save_freq=4000 --log_freq=500 --wandb.enable=false
ACTF=outputs/train/act_fixed/checkpoints/last/pretrained_model
run ch08_gen_fixed_A   python ch06/eval_act.py --checkpoint $ACTF --dataset-root $DSF --repo-id physicalai/so101_fixed_sim --episodes 10 --cube 0.24 0.0
run ch08_gen_fixed_B   python ch06/eval_act.py --checkpoint $ACTF --dataset-root $DSF --repo-id physicalai/so101_fixed_sim --episodes 10 --cube 0.22 0.08
run ch08_gen_fixed_C   python ch06/eval_act.py --checkpoint $ACTF --dataset-root $DSF --repo-id physicalai/so101_fixed_sim --episodes 10 --cube 0.20 -0.10
run ch08_gen_fixed_R   python ch06/eval_act.py --checkpoint $ACTF --dataset-root $DSF --repo-id physicalai/so101_fixed_sim --episodes 20
run ch08_gen_rand_A    python ch06/eval_act.py --checkpoint $ACT --episodes 10 --cube 0.24 0.0
run ch08_gen_rand_B    python ch06/eval_act.py --checkpoint $ACT --episodes 10 --cube 0.22 0.08
run ch08_gen_rand_C    python ch06/eval_act.py --checkpoint $ACT --episodes 10 --cube 0.20 -0.10

# 8-1
DSM=outputs/datasets/so101_multicolor_sim
run ch08_record_multi  python ch08/record_multicolor.py --episodes 60 --root $DSM
run ch08_train_multi   lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=physicalai/so101_multicolor_sim --dataset.root=$DSM \
    --rename_map="$RENAME" --policy.device=cuda --policy.push_to_hub=false \
    --output_dir=outputs/train/smolvla_multicolor --job_name=smolvla_multicolor --steps=$VLA_STEPS --batch_size=8 \
    --save_freq=2500 --log_freq=250 --wandb.enable=false
run ch08_eval_multi    python ch08/eval_multicolor.py --checkpoint outputs/train/smolvla_multicolor/checkpoints/last/pretrained_model --episodes-per-color 10 --video
echo "done $(date +%H:%M:%S)"
