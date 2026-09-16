"""2부 실습 — MuJoCo 첫 실행

XML 문자열 하나로 가상 세계를 만들고, 물리 시뮬레이션을 몇 스텝 돌려 봅니다.
공중에 놓인 공(sphere)이 중력으로 떨어지는 가장 단순한 세계입니다.

실행:
    python ch02/mujoco_first.py
"""

import mujoco

# MJCF(MuJoCo XML) — 바닥(plane) 하나와 공 하나
WORLD_XML = """
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="ball" pos="0 0 1.0">
      <freejoint/>
      <geom type="sphere" size="0.05" mass="0.1" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_string(WORLD_XML)   # 세계의 '설계도'
    data = mujoco.MjData(model)                         # 세계의 '현재 상태'

    print(f"MuJoCo v{mujoco.__version__}")
    print(f"timestep = {model.opt.timestep} s  (한 스텝에 흐르는 시간)")
    print(f"body 수 = {model.nbody}, geom 수 = {model.ngeom}, 자유도(nq) = {model.nq}")
    print()

    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    mujoco.mj_forward(model, data)           # 초기 상태에서 위치 등 파생값을 한 번 계산

    print(" time(s)   ball z(m)")
    for step in range(1001):
        if step % 100 == 0:
            z = data.xpos[ball_id][2]        # xpos: 각 body의 월드 좌표 위치
            print(f"{data.time:7.3f}   {z:8.4f}")
        mujoco.mj_step(model, data)          # 물리 세계를 timestep 만큼 진행
