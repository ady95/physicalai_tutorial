"""SO-101 로봇팔 + 책상 + 블록 장면(MJCF)을 만드는 도우미.

3부부터 8부까지 같은 장면을 조금씩 바꿔 가며 사용합니다.
Menagerie의 so101.xml 을 include 해야 하는데, MuJoCo는 include 경로와 mesh 경로를
"메인 XML 파일이 있는 폴더" 기준으로 찾습니다. 그래서 장면 XML을 문자열로 만든 뒤
so101.xml 과 같은 폴더에 파일로 써 놓고 그 경로를 돌려줍니다.
"""

import os

from common.menagerie import so101_xml

SCENE_TEMPLATE = """
<mujoco model="physicalai_scene">
  <include file="so101.xml"/>

  <option timestep="0.002" integrator="implicitfast"/>
  <visual>
    <global offwidth="640" offheight="480"/>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>

  <asset>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.85 0.85 0.85"
             rgb2="0.75 0.75 0.75" markrgb="0.6 0.6 0.6" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="8 8" reflectance="0.1"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="1 1 0.05" material="groundplane"/>

    <!-- 집을 물체: 빨간 블록 (한 변 3 cm) -->
    <body name="cube" pos="{cube_x} {cube_y} 0.015">
      <freejoint name="cube_free"/>
      <geom name="cube_geom" type="box" size="0.015 0.015 0.015" mass="0.02"
            rgba="0.9 0.1 0.1 1" friction="1.5 0.005 0.0001" condim="4"/>
    </body>

    <!-- 놓을 곳: 상자 (바닥 + 낮은 벽 4개) -->
    <body name="box" pos="{box_x} {box_y} 0">
      <geom type="box" size="0.05 0.05 0.003" pos="0 0 0.003" rgba="0.2 0.4 0.9 1"/>
      <geom type="box" size="0.05 0.003 0.015" pos="0 0.05 0.015" rgba="0.2 0.4 0.9 1"/>
      <geom type="box" size="0.05 0.003 0.015" pos="0 -0.05 0.015" rgba="0.2 0.4 0.9 1"/>
      <geom type="box" size="0.003 0.05 0.015" pos="0.05 0 0.015" rgba="0.2 0.4 0.9 1"/>
      <geom type="box" size="0.003 0.05 0.015" pos="-0.05 0 0.015" rgba="0.2 0.4 0.9 1"/>
    </body>

    <!-- 카메라 3대: 위에서, 옆에서, 정면에서 -->
    <camera name="top" pos="0.2 0 0.8" quat="1 0 0 0"/>
    <camera name="side" mode="targetbody" target="cube" pos="0.55 -0.55 0.35"/>
    <camera name="front" mode="targetbody" target="cube" pos="0.75 0 0.3"/>
  </worldbody>

  <keyframe>
    <!-- 대기 자세: 팔을 접어 손을 16 cm 높이에 두어 바닥의 물체를 건드리지 않게 -->
    <key name="home" qpos="0 -1.57 1.57 0 0 0 {cube_x} {cube_y} 0.015 1 0 0 0" ctrl="0 -1.57 1.57 0 0 0"/>
  </keyframe>
</mujoco>
"""


def build_scene(cube_pos=(0.25, 0.0), box_pos=(0.0, 0.25), filename="physicalai_scene.xml") -> str:
    """장면 XML 파일을 so101.xml 옆에 쓰고 경로를 돌려줍니다."""
    robot_dir = os.path.dirname(so101_xml("so101.xml"))
    xml = SCENE_TEMPLATE.format(cube_x=cube_pos[0], cube_y=cube_pos[1],
                                box_x=box_pos[0], box_y=box_pos[1])
    path = os.path.join(robot_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml)
    return path
