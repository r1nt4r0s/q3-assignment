"""HuggingFace-trained issue classification -> Genesis mobile inspection rover.

Physical: wheel-joint velocity commands in Genesis, A* waypoints, pose feedback.
Perception: *external virtual roadside cameras* provide held-out HF photographs.
Images are NOT pixels rendered by the Genesis camera. The 3D overhead camera is
for visualization only. This is a coupled benchmark, not sim2real validation.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import json
import math
import numpy as np
import pandas as pd
from road_ml import RoadVision, select_station, safe_speed_limit, label_risk
from genesis_mission import make_scene, xy_yaw, MissionSettings
from pai_patrol import BASE, STATIONS, FloorMap

@dataclass
class VisionSettings(MissionSettings):
    # 20 physics steps/sec, 20 recorded fps: nominal MP4 runtime should be 90 s
    fps:int = 20
    duration_s:float = 90.
    dt:float = .05
    policy:str = 'adaptive'


def load_virtual_camera_feeds(checkpoint, manifest_path):
    model = RoadVision(checkpoint)
    manifest = json.loads(Path(manifest_path).read_text(encoding = 'utf8'))
    feeds = {}
    for entry in manifest:
        station = entry['station']
        if station not in STATIONS:
            raise ValueError('Unrecognized station '+station)
        image_path = Path(manifest_path).parent/entry['image']
        prediction = model.predict(image_path)
        feeds[station] = {'path':str(image_path.resolve()),'truth_category':entry['label'], **prediction}
    if set(feeds) != set(STATIONS):
        raise ValueError('Need one independent HF held-out image per station A..F')
    return feeds


def run_visual_mission(settings, feeds, output, ppo = None):
    import cv2
    output = Path(output)
    output.mkdir(parents = True, exist_ok = True)
    scene, rover, camera, joints = make_scene(settings, output)
    floor = FloorMap()
    visited = set()
    target = None
    route = []
    idx = 0
    risks = {k:feeds[k]['policy_risk'] for k in STATIONS}
    scores = risks if settings.policy == 'adaptive' else {k:0 for k in STATIONS}
    filename=output/f'genesis_visual_{settings.policy}.mp4'
    trace = []
    visits = []
    last_xy = None
    travel = 0.
    first_high_visit = None
    camera.start_recording(save_to_filename = str(filename.resolve()), fps = settings.fps)
    try:
        for step in range(int(settings.duration_s/settings.dt)):
            t = step*settings.dt
            pos, yaw, z = xy_yaw(rover)
            if last_xy is not None:
                travel += math.dist(pos, last_xy)
            last_xy = pos
            if target is None:
                target = select_station(pos, STATIONS, scores, visited, settings.policy)
                if target is not None:
                    route = floor.plan(pos, STATIONS[target])
                    idx = 0
            v = omega = 0.
            local_risk = 0.
            if target is not None:
                distance = math.dist(pos, STATIONS[target])
                local_risk = risks[target]*max(0., 1.-distance/3.)
                if distance<.45:
                    visited.add(target)
                    visits.append({'station':target,'arrival_s':t,'groundtruth_category':feeds[target]['truth_category'],
                                   'predicted_category':feeds[target]['label'],'risk_policy':risks[target]})
                    if label_risk(feeds[target]['truth_category']) >= .8 and first_high_visit is None:
                        first_high_visit = t
                    target = None
                    route = []
                else:
                    if idx >= len(route):
                        route = floor.plan(pos, STATIONS[target])
                        idx = 0
                    if idx<len(route)-1 and math.dist(pos, route[idx])<.27:
                        idx += 1
                    wp = route[idx]
                    angle = math.atan2(wp[1]-pos[1], wp[0]-pos[0])
                    error = (angle-yaw+math.pi)%(2*math.pi)-math.pi
                    omega = float(np.clip(2.0*error, -settings.max_omega, settings.max_omega))
                    max_v = safe_speed_limit(local_risk, settings.max_v)
                    if ppo is not None:
                        obs = np.array([local_risk, 0., min(1., travel/18.), min(1., t/settings.duration_s)], dtype = np.float32)
                        action, _ = ppo.predict(obs, deterministic = True)
                        max_v = min(max_v, float(np.clip(np.asarray(action).reshape(-1)[0], 0, 1))*settings.max_v)
                    v = max_v*max(0., 1.-abs(error)/.8)
            wl = settings.wheel_sign*(v-.5*settings.wheel_track*omega)/settings.wheel_radius
            wr = settings.wheel_sign*(v+.5*settings.wheel_track*omega)/settings.wheel_radius
            rover.control_dofs_velocity(np.asarray([wl, wr]), dofs_idx_local = joints)
            scene.step()
            trace.append({'time_s':t,'x':pos[0],'y':pos[1],'z':z,'target':target or 'NONE',
                          'risk_from_hf':local_risk,'speed_cmd_mps':v,'visited':len(visited)})
    finally:
        try:
            rover.control_dofs_velocity(np.zeros(2), dofs_idx_local = joints)
        except Exception:
            pass
        camera.stop_recording()
    pd.DataFrame(trace).to_csv(output/f'visual_{settings.policy}_physical_trace.csv',index=False)
    pd.DataFrame(visits).to_csv(output/f'visual_{settings.policy}_visits.csv',index=False)
    metrics={'policy':settings.policy,'true_physical_distance_m':travel,
             'visited':len(visited),'first_high_risk_visit_s':first_high_visit,
             'duration_sim_s':settings.duration_s,'video':str(filename),
             'description':'External virtual roadside photo feed -> HF-trained classification -> Genesis real physics motion',
             'not_an_onboard_genesis_camera':True}
    (output/f'visual_{settings.policy}_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
    # A vital assertion that file runtime matches 1-3 min assignment window.
    cap = cv2.VideoCapture(str(filename))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    video_seconds = n/fps if fps>0 else 0
    metrics['observed_mp4_seconds'] = video_seconds
    if not (60 <= video_seconds <= 180):
        raise RuntimeError(f'Physical run succeeded but MP4 duration={video_seconds:.1f}s. Re-encode video for 60-180s; check Genesis recorded FPS.')
    return metrics, pd.DataFrame(trace), filename
