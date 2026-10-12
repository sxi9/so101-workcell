"""Generate assets/workcell/so101_workcell.xml from the upstream SO-101 MJCF.

Changes relative to upstream (TheRobotStudio so101_new_calib.xml):
  * box collision pads on both fingertips (mesh-mesh contacts are unstable for grasping)
  * a wrist camera on the gripper body
  * meshdir pointing at the shared upstream asset folder

Run: python scripts/build_workcell_xml.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "so101_workcell/sim/assets/so101/so101_new_calib.xml"
DST = ROOT / "so101_workcell/sim/assets/workcell/so101_workcell.xml"

# Fingertip pads, positions measured from the mesh vertices at the home pose
# (see scripts/measure_geometry.py). Frames: gripper body and moving_jaw body.
FIXED_PAD = (
    '        <geom name="pad_fixed" class="pad" type="box" pos="-0.0115 0 -0.096" '
    'size="0.0045 0.006 0.009"/>\n'
)
MOVING_PAD = (
    '        <geom name="pad_moving" class="pad" type="box" pos="-0.0105 -0.076 0.019" '
    'size="0.0045 0.009 0.006"/>\n'
)
WRIST_CAM = (
    '        <camera name="wrist" pos="0 0.035 -0.01" xyaxes="1 0 0 0 1 0.3" fovy="75"/>\n'
)
PAD_DEFAULT = """    <default class="pad">
      <geom group="3" contype="1" conaffinity="1" condim="4" friction="1.5 0.02 0.001"
            solref="0.004 1" solimp="0.95 0.99 0.001" rgba="0.1 0.1 0.1 1"/>
    </default>
"""


def main() -> None:
    xml = SRC.read_text()
    xml = xml.replace('meshdir="assets"', 'meshdir="../so101/assets"')
    xml = xml.replace('<mujoco model="so101_new_calib">', '<mujoco model="so101_workcell">')
    # pad default class goes inside the first <default> block
    xml = xml.replace("  <default>\n", "  <default>\n" + PAD_DEFAULT, 1)
    # fixed pad + wrist camera right after the gripperframe site
    xml = re.sub(
        r'(\s*<site group="3" name="gripperframe"[^>]*/>\n)',
        lambda m: m.group(1) + FIXED_PAD + WRIST_CAM,
        xml,
        count=1,
    )
    # moving pad after the moving jaw's inertial element
    xml = re.sub(
        r'(<body name="moving_jaw_so101_v1"[^>]*>\n(?:.*\n)*?\s*<inertial[^>]*/>\n)',
        lambda m: m.group(1) + MOVING_PAD,
        xml,
        count=1,
    )
    # Finger meshes collide as convex hulls, which are much fatter than the real jaws and hit
    # the object before the pads do. Make every mesh collision geom from the gripper body
    # onward visual-only; the pads are the only contact surfaces on the fingers.
    head, tail = xml.split('<body name="gripper"', 1)
    tail = tail.replace('class="collision"', 'class="collision" contype="0" conaffinity="0"')
    xml = head + '<body name="gripper"' + tail
    assert "pad_fixed" in xml and "pad_moving" in xml and 'name="wrist"' in xml
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(xml)
    print("wrote", DST.relative_to(ROOT))


if __name__ == "__main__":
    main()
