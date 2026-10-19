"""Genesis World 3-D rigid-body mobile rover driven by wheel velocity commands.

All observations are synthetic; the simulator is a *simulated* factory. No hardware validation.
The policy receives only synthetic sensor observations, never ground-truth fault IDs/times.
3-D dynamic motion is separately logged; planner-only benchmark is a DIFFERENT experiment.
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from pai_patrol import (BASE, STATIONS, OBSTACLES, STATION_KEYS, CFG,
                        Detector, Patrol, FloorMap, make_scenario, Scenario)

@dataclass
class MissionSettings:
    dt: float = 0.05
    duration_s: float = 90.0
    fps: int = 10
    backend: str = 'cuda'  # 'cpu' option for diagnostics
    res: tuple[int, int] = (640, 480)
    policy: str = 'adaptive'  # 'nearest' or 'adaptive'
    seed: int = 42
    max_v: float = 0.65
    max_omega: float = 1.9
    wheel_radius: float = 0.12
    wheel_track: float = 0.41
    # URDF positive y wheel axes rotate the rover toward -x for positive omega_y
    wheel_sign: float = -1.0


def generate_rover_urdf(path: Path):
    """Create a simple explicitly inertialized differential drive URDF, no external assets."""
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    text = '''<?xml version="1.0"?>
<robot name="pai_differential_rover">
 <material name="blue"><color rgba="0.13 0.39 0.78 1"/></material>
 <material name="dark"><color rgba="0.12 0.15 0.20 1"/></material>
 <material name="accent"><color rgba="0.95 0.65 0.20 1"/></material>
 <link name="base_link">
   <inertial><origin xyz="0 0 0"/><mass value="6"/><inertia ixx="0.14" ixy="0" ixz="0" iyy="0.18" iyz="0" izz="0.20"/></inertial>
   <visual><geometry><box size="0.50 0.32 0.13"/></geometry><material name="blue"/></visual>
   <collision><geometry><box size="0.50 0.32 0.13"/></geometry></collision>
 </link>
 <link name="left_wheel">
  <inertial><origin xyz="0 0 0"/><mass value="0.5"/><inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/></inertial>
  <visual><origin rpy="1.57079632679 0 0"/><geometry><cylinder radius="0.12" length="0.06"/></geometry><material name="dark"/></visual>
  <collision><origin rpy="1.57079632679 0 0"/><geometry><cylinder radius="0.12" length="0.06"/></geometry></collision>
 </link>
 <joint name="left_wheel_joint" type="continuous">
  <parent link="base_link"/><child link="left_wheel"/><origin xyz="-0.05 0.205 -0.13" rpy="0 0 0"/>
  <axis xyz="0 1 0"/><limit effort="20" velocity="25"/><dynamics damping="0.03" friction="0.01"/>
 </joint>
 <link name="right_wheel">
  <inertial><origin xyz="0 0 0"/><mass value="0.5"/><inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/></inertial>
  <visual><origin rpy="1.57079632679 0 0"/><geometry><cylinder radius="0.12" length="0.06"/></geometry><material name="dark"/></visual>
  <collision><origin rpy="1.57079632679 0 0"/><geometry><cylinder radius="0.12" length="0.06"/></geometry></collision>
 </link>
 <joint name="right_wheel_joint" type="continuous">
  <parent link="base_link"/><child link="right_wheel"/><origin xyz="-0.05 -0.205 -0.13" rpy="0 0 0"/>
  <axis xyz="0 1 0"/><limit effort="20" velocity="25"/><dynamics damping="0.03" friction="0.01"/>
 </joint>
 <link name="caster_ball">
  <inertial><origin xyz="0 0 0"/><mass value="0.1"/><inertia ixx="0.00015" ixy="0" ixz="0" iyy="0.00015" iyz="0" izz="0.00015"/></inertial>
  <visual><geometry><sphere radius="0.06"/></geometry><material name="accent"/></visual>
  <collision><geometry><sphere radius="0.06"/></geometry></collision>
 </link>
 <joint name="caster_joint" type="fixed"><parent link="base_link"/><child link="caster_ball"/><origin xyz="0.19 0 -0.19"/></joint>
</robot>'''
    path.write_text(text, encoding = 'utf8')
    return path


def as_numpy(t):
    return t.detach().cpu().numpy() if hasattr(t, 'detach') else np.asarray(t)


def xy_yaw(robot):
    xyz = as_numpy(robot.get_pos()).reshape(-1, 3)[0]
    w, x, y, z = as_numpy(robot.get_quat()).reshape(-1, 4)[0]
    yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return (float(xyz[0]), float(xyz[1])), float(yaw), float(xyz[2])


def make_scene(settings:MissionSettings, output:Path):
    import genesis as gs
    if settings.dt <= 0 or settings.duration_s <= 0:
        raise ValueError('Invalid time settings')
    output = Path(output)
    output.mkdir(parents = True, exist_ok = True)
    scene=gs.Scene(show_viewer=False,
                   sim_options = gs.options.SimOptions(dt = settings.dt),
                   vis_options = gs.options.VisOptions(show_world_frame = False))
    scene.add_entity(gs.morphs.Plane(), surface = gs.surfaces.Plastic(color = (.80, .84, .88, 1)))
    # Ground obstacles: rectangles come from same grid as the planner.
    for i, (x1, y1, x2, y2) in enumerate(OBSTACLES):
        scene.add_entity(gs.morphs.Box(pos=((x1+x2)/2,(y1+y2)/2,.45),
                                      size = (x2-x1, y2-y1, .9), fixed = True),
                         surface=gs.surfaces.Plastic(color=(.38,.42,.48,1)), name=f'obstacle_{i}')
    for k, (x, y) in STATIONS.items():
        # Device cabinets are deliberately offset from collision-free inspection points.
        scene.add_entity(gs.morphs.Box(pos=(x+.58,y+.50,.36),
                                      size = (.42, .34, .72), fixed = True),
                         surface=gs.surfaces.Plastic(color=(.16,.50,.45,1)), name=f'equipment_{k}')
        scene.add_entity(gs.morphs.Box(pos=(x,y,.012),size=(.22,.22,.024),fixed=True,collision=False),
                         surface=gs.surfaces.Plastic(color=(.98,.76,.12,1)), name=f'inspect_marker_{k}')
    urdf = generate_rover_urdf(output/'pai_rover.urdf')
    rover=scene.add_entity(gs.morphs.URDF(file=str(urdf.resolve()),
                                           pos = (BASE[0], BASE[1], .25),
                                           fixed = False, merge_fixed_links = True),
                           name = 'pai_rover')
    camera=scene.add_camera(res=settings.res,
                            pos = (15.8, -13.0, 17.8), lookat = (6.0, 4.2, .0), fov = 48, GUI = False)
    scene.build()
    indices = [int(rover.get_joint(n).dofs_idx_local[0]) for n in ('left_wheel_joint','right_wheel_joint')]
    rover.set_dofs_kp(np.zeros(2), indices)
    rover.set_dofs_kv(np.array([7., 7.]), indices)
    rover.set_dofs_force_range(lower = np.array([-18., -18.]), upper = np.array([18., 18.]), dofs_idx_local = indices)
    return scene, rover, camera, indices


def run_3d(settings:MissionSettings, detector:Detector, output:Path, smoke = False):
    """3D wheel-actuated closed-loop run. Returns physical metrics, separate from 2D benchmark."""
    scene, rover, camera, joints = make_scene(settings, output)
    scenario = make_scenario(settings.seed)
    p = Patrol(scenario, detector, policy = settings.policy)
    phys_rows = []
    events = []
    visits = []
    path=Path(output)/('genesis_smoke.mp4' if smoke else f'genesis_{settings.policy}.mp4')
    total_s = 2.0 if smoke else settings.duration_s
    steps = round(total_s/settings.dt)
    last_xy = None
    physical_dist = 0.
    boundaries = 0
    ground_contact_flags = 0
    # Ground truth stays in `scenario`; Patrol only uses it in evaluation and onsite checks.
    camera.start_recording(save_to_filename = str(path.resolve()), fps = settings.fps)
    try:
        for i in range(steps):
            p.t = i*settings.dt
            pos, theta, z = xy_yaw(rover)
            p.pos = np.array(pos)
            p.heading = theta
            if last_xy is not None:
                physical_dist += math.dist(pos, last_xy)
            last_xy = pos
            if p.floor.blocked(*pos, margin = .12):
                boundaries += 1 # geofence/proximity proxy, NOT a measured collision impulse
            if z<-.05:
                ground_contact_flags += 1
            if not smoke:
                if p.t + 1e-6 >= p.next_remote:
                    p.observe_remote()
                    p.next_remote += p.cfg.remote_period
                    if settings.policy == 'adaptive' and p.target not in (None,'HOME') and p.t-p.target_start>6:
                        choices = p.options()
                        serious=[k for k in choices if k!=p.target and
                                p.consecutive_alarm[k] >= 2 and
                                p.remote_risks[k]>p.remote_risks.get(p.target, 0)+1.05]
                        if serious:
                            best = max(serious, key = lambda k:p.remote_risks[k]-.18*p.floor.distance(p.pos, STATIONS[k]))
                            p.set_target(best)
                if p.target is None and not p.finished:
                    p.set_target(p.choose_target())
            else:
                if p.target is None:
                    p.set_target('A')
            v = 0.
            omega = 0.
            if p.target is not None and not p.finished:
                target = np.asarray(BASE if p.target == 'HOME' else STATIONS[p.target])
                if np.linalg.norm(target-p.pos)<.40:
                    if smoke:
                        p.finished = True
                    else:
                        p.arrive()
                if not p.finished and p.target is not None:
                    if p.waypoint >= len(p.route):
                        p.route = p.floor.plan(p.pos, target)
                        p.waypoint = 0
                        p.replans += 1
                    wp = np.asarray(p.route[p.waypoint])
                    if np.linalg.norm(wp-p.pos)<.28 and p.waypoint<len(p.route)-1:
                        p.waypoint += 1
                        wp = np.asarray(p.route[p.waypoint])
                    error = (math.atan2(wp[1]-p.pos[1], wp[0]-p.pos[0])-theta+math.pi)%(2*math.pi)-math.pi
                    omega = float(np.clip(2.2*error, -settings.max_omega, settings.max_omega))
                    v = float(settings.max_v*max(0, 1-abs(error)/.80))
            # Twist -> joint velocities. Sign is determined by URDF wheel joint axes.
            wl = settings.wheel_sign*(v-settings.wheel_track*omega/2)/settings.wheel_radius
            wr = settings.wheel_sign*(v+settings.wheel_track*omega/2)/settings.wheel_radius
            rover.control_dofs_velocity(np.asarray([wl, wr]), dofs_idx_local = joints)
            scene.step()
            phys_rows.append(dict(t=p.t,x=pos[0],y=pos[1],z=z,yaw=theta,
                                  v_command = v, omega_command = omega,
                                  target = p.target or 'NONE',
                                  max_ai_risk = max(p.remote_risks.values()),
                                  completed = p.finished))
            if p.finished and not smoke:
                # Continue recording a stationary robot to preserve a 60-180s video.
                rover.control_dofs_velocity(np.zeros(2), dofs_idx_local = joints)
        # Recording is also stopped in finally.
    finally:
        try:
            rover.control_dofs_velocity(np.zeros(2), dofs_idx_local = joints)
        except Exception:
            pass
        camera.stop_recording()
    phys = pd.DataFrame(phys_rows)
    phys.to_csv(Path(output)/'genesis_physical_trace.csv', index = False)
    pd.DataFrame(p.visits).to_csv(Path(output)/'genesis_visits.csv', index = False)
    pd.DataFrame(p.history).to_csv(Path(output)/'genesis_ai_risks.csv', index = False)
    # Note: p.events is the policy control event log; 3D distance derived from rigid body position.
    (Path(output)/'genesis_events.json').write_text(json.dumps(p.events, indent = 2, ensure_ascii = False))
    result={'policy':settings.policy,'seed':settings.seed,'simulated_3d_seconds':total_s,
            'physical_distance_m':physical_dist,'inspected_stations':len(p.visits),
            'replans':p.replans,'geofence_proximity_steps_proxy':boundaries,
            'below_floor_steps':ground_contact_flags,'video':str(path),
            'method':'Genesis wheel-joint velocity control and base pose feedback; synthetic telemetry',
            'warning':'Not a real-world robot experiment; geofence is not a contact sensor.'}
    (Path(output)/'genesis_physical_metrics.json').write_text(json.dumps(result, indent = 2, ensure_ascii = False))
    return result, phys, path
