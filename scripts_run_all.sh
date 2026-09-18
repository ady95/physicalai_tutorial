#!/usr/bin/env bash
# 책의 모든 실습 스크립트를 순서대로 실행하고 로그를 outputs/logs/ 에 남긴다 (집필 검증용).
set -uo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export MUJOCO_GL=egl
mkdir -p outputs/logs
F='EGLError|glCheckError|Warning|warn'
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
run ch01_agent_3lines      python ch01/agent_3lines.py
run ch01_random_agent      python ch01/random_agent.py
run ch02_check_env         python ch02/check_env.py
run ch02_mujoco_first      python ch02/mujoco_first.py
run ch02_drop_cube         python ch02/drop_cube.py
run ch02_drop_cube_moon    python ch02/drop_cube.py --gravity -1.62 --no-video
run ch02_robot_arm_joints  python ch02/robot_arm_joints.py
run ch03_read_pose         python ch03/read_pose.py
run ch03_fk_2link          python ch03/fk_2link.py
run ch03_ik_reach          python ch03/ik_reach.py
run ch03_pick_and_place    python ch03/pick_and_place.py
run ch03_pnp_trials        python ch03/pick_and_place.py --no-video --trials 10
run ch04_opencv_basics     python ch04/opencv_basics.py
run ch04_color_detect      python ch04/color_detect.py
run ch04_sim_camera        python ch04/sim_camera.py
run ch04_find_cube         python ch04/find_cube_in_camera.py
run ch04_yolo              python ch04/yolo_detect.py
run ch04_vision_pnp        python ch04/vision_pick_and_place.py
run ch04_vision_trials     python ch04/vision_pick_and_place.py --no-video --trials 10
if [ "${WITH_RL:-0}" = "1" ]; then
  run ch05_point_dense     python ch05/train_point_reach.py
  run ch05_point_sparse    python ch05/train_point_reach.py --reward sparse
  for r in dense sparse shaped; do run ch05_arm_$r python ch05/train_arm_reach.py --reward $r --steps 300000; done
  run ch05_arm_eval        python ch05/train_arm_reach.py --eval-only --reward dense
fi
echo "done: $(ls outputs/logs | wc -l) logs"
