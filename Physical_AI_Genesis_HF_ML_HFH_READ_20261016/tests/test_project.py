"""Tests runnable without importing Genesis World (Colab-only dependency)."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from pai_patrol import smoke_test, Detector, Patrol, make_scenario, FloorMap, STATIONS, BASE
from genesis_mission import generate_rover_urdf

def test_ai_smoke():
    x = smoke_test()
    assert x.metrics['elapsed_s']>0
    assert x.metrics['recall'] >= 0

def test_planner_accessibility():
    floor = FloorMap()
    for end in STATIONS.values():
        assert len(floor.plan(BASE, end))>0

def test_urdf(tmp_path):
    f = generate_rover_urdf(tmp_path/'rover.urdf')
    tree = ET.parse(f)
    assert len(tree.findall('.//link')) == 4
    joints = tree.findall('.//joint')
    assert {j.get('name') for j in joints} == {'left_wheel_joint','right_wheel_joint','caster_joint'}
    assert all(j.find('axis') is not None for j in joints if j.get('type') == 'continuous')
