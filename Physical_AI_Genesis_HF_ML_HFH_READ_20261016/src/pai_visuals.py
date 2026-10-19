"""Visuals and MP4 rendering for Physical AI patrol demo.
All overlays are in English to avoid missing Japanese fonts on free Colab.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import imageio.v2 as imageio
from IPython.display import display, Video
from pai_patrol import BASE, WIDTH, HEIGHT, STATIONS, STATION_KEYS, OBSTACLES, CFG

CANVAS_W, CANVAS_H = 580, 530
BLUE = (205, 140, 55)
RED = (70, 80, 228)
GREEN = (65, 170, 100)
INK = (34, 37, 45)

def risk_chart(result, station_order = STATION_KEYS):
    h = pd.DataFrame(result.remote_history)
    fig, ax = plt.subplots(figsize = (10, 3.6))
    for name in station_order:
        ax.plot(h['t'], h[name], label = name, linewidth = 1.5)
    ax.axhline(CFG.risk_gate, ls = '--', color = 'black', alpha = .5, label = 'alert threshold')
    ax.set(xlabel='Time (s)',ylabel='Remote AI risk (normalized)',title=f'{result.policy}: remote telemetry -> anomaly risk')
    ax.legend(ncol = 7, loc = 'upper center', bbox_to_anchor = (.5, -.2))
    ax.grid(alpha = .2)
    fig.tight_layout()
    return fig

def benchmark_charts(df, deltas):
    metrics=[('recall','Event detection recall','higher'),('weighted_detection_delay_s','Weighted detection delay (s)','lower'),
             ('distance_m','Travel distance (m)','lower')]
    mean = df.groupby('policy').mean(numeric_only = True)
    fig, axs = plt.subplots(1, 3, figsize = (12, 3.8))
    for ax, (col, title, better) in zip(axs, metrics):
        vals = [float(mean.loc['nearest', col]), float(mean.loc['adaptive', col])]
        bars = ax.bar(['Distance-only','AI-adaptive'], vals, color = ['#9ca3af','#2563eb'])
        ax.set_title(title, fontsize = 10)
        ax.set_ylim(bottom = 0)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,v,f'{v:.2f}',ha='center',va='bottom',fontsize=9)
        ax.tick_params(axis = 'x', labelrotation = 17)
    fig.suptitle(f'Paired evaluation across {len(deltas)} seeded scenarios (simulation)',fontsize=12)
    fig.tight_layout()
    return fig

def _text(img, text, x, y, size = .54, color = INK, weight = 1):
    cv2.putText(img, str(text), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, weight, cv2.LINE_AA)

def _point(pt, xoff = 0, yoff = 62):
    return (round(pt[0]/WIDTH*535+22+xoff), round((HEIGHT-pt[1])/HEIGHT*396+yoff))

def _panel(result, t):
    img = np.full((CANVAS_H, CANVAS_W, 3), 245, dtype = np.uint8)
    adaptive = result.policy == 'adaptive'
    _text(img,'AI risk-aware planner' if adaptive else 'Distance-only baseline', 18, 27, .70, (43, 85, 170) if adaptive else INK, 2)
    _text(img,'Virtual plant / 2D differential-drive', 18, 47, .43, (93, 97, 105))
    cv2.rectangle(img, _point((0, HEIGHT)), _point((WIDTH, 0)), (225, 226, 230), 2)
    for obs in OBSTACLES:
        a, b, c, d = obs
        cv2.rectangle(img, _point((a, d)), _point((c, b)), (162, 163, 169), -1)
        cv2.rectangle(img, _point((a, d)), _point((c, b)), (119, 121, 127), 2)
    latest = [v for v in result.visits if v['t'] <= t]
    statuses = {v['station']:v for v in latest}
    for key, pt in STATIONS.items():
        xy = _point(pt)
        state = statuses.get(key)
        color = RED if state and state['flagged'] else (GREEN if state else BLUE)
        cv2.circle(img, xy, 14, color, -1)
        cv2.circle(img, xy, 14, INK, 1)
        _text(img, key, xy[0]-6, xy[1]+5, .47, (255, 255, 255), 2)
    home = _point(BASE)
    cv2.rectangle(img, (home[0]-11, home[1]-11), (home[0]+11, home[1]+11), (75, 75, 75), -1)
    _text(img,'H', home[0]-6, home[1]+5, .44, (255, 255, 255), 1)
    track = result.trajectory
    index = min(len(track)-1, max(0, int(t/CFG.dt)))
    current = track[index]
    coords = np.array([_point((p[1], p[2])) for p in track[:index+1]], dtype = np.int32)
    if len(coords)>1:
        cv2.polylines(img, [coords], False, (170, 95, 28) if adaptive else (70, 130, 75), 2, cv2.LINE_AA)
    _, rx, ry, heading, target, maxrisk, alerts = current
    center = _point((rx, ry))
    v1 = (round(center[0]+15*math.cos(heading)), round(center[1]-15*math.sin(heading)))
    v2 = (round(center[0]+11*math.cos(heading+2.45)), round(center[1]-11*math.sin(heading+2.45)))
    v3 = (round(center[0]+11*math.cos(heading-2.45)), round(center[1]-11*math.sin(heading-2.45)))
    cv2.fillConvexPoly(img, np.array([v1, v2, v3], np.int32), (40, 49, 61))
    cv2.circle(img, center, 15, (255, 255, 255), 1)
    basey = 486
    cv2.line(img, (14, 470), (565, 470), (187, 189, 192), 1)
    traveled = sum(math.dist((track[i-1][1], track[i-1][2]), (track[i][1], track[i][2])) for i in range(1, index+1))
    _text(img,f't={t:05.1f}s  Target={str(target):4s}  Alerts={int(alerts)}/2',17,basey,.49,INK,1)
    _text(img,f'Travel={traveled:05.1f}m  Max ML risk={maxrisk:.2f}',17,508,.47,(75,78,86),1)
    return img

def _intro_frame(heading, sub1, sub2):
    h, w = 560, 1200
    a = np.full((h, w, 3), (25, 29, 40), dtype = np.uint8)
    _text(a, heading, 75, 190, 1.17, (255, 255, 255), 2)
    _text(a, sub1, 77, 258, .73, (175, 200, 245), 1)
    _text(a, sub2, 77, 311, .63, (180, 184, 190), 1)
    return a

def render_comparison_video(baseline, adaptive, path = 'outputs/physical_ai_patrol.mp4', fps = 10, opening_s = 4, outro_s = 7, quiet = True):
    """Side-by-side actual computed trajectories; encode H264. Frames correspond to simulation timestamps."""
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    end_t = max(baseline.metrics['elapsed_s'], adaptive.metrics['elapsed_s'])
    duration = opening_s+end_t+outro_s
    if duration<60:
        # Freeze the finished scene rather than misrepresent simulation time or speed.
        outro_s += int(math.ceil(61-duration))
        duration = opening_s+end_t+outro_s
    if duration>180:
        raise ValueError('Video exceeds 3 minute submission limit; shorten selected run.')
    options = dict(fps = fps, codec = 'libx264', quality = 7, macro_block_size = 16)
    with imageio.get_writer(str(path), format = 'FFMPEG', **options) as writer:
        title=_intro_frame('PHYSICAL AI  |  PLANT INSPECTION',
            'Isolation Forest + dynamic risk-based routing',
            'Comparative simulation: same faults / same sensors / paired seed')
        for _ in range(int(opening_s*fps)):
            writer.append_data(cv2.cvtColor(title, cv2.COLOR_BGR2RGB))
        for f in range(int(math.ceil(end_t*fps))):
            t = f/fps
            p1 = _panel(baseline, t)
            p2 = _panel(adaptive, t)
            canvas = np.full((560, 1200, 3), 255, dtype = np.uint8)
            canvas[8:538, 10:590] = p1
            canvas[8:538, 610:1190] = p2
            _text(canvas,'Simulation  |  synthetic plant sensors  |  1.0x playback', 12, 552, .4, (60, 63, 71))
            writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            if (f%200 == 0 and not quiet):
                print('encoded', f,'/', int(end_t*fps))
        end=_intro_frame('COMPARISON RESULTS',
            f"Recall: baseline {baseline.metrics['recall']:.0%} / AI {adaptive.metrics['recall']:.0%}",
            f"Weighted detection delay: {baseline.metrics['weighted_detection_delay_s']:.1f}s / {adaptive.metrics['weighted_detection_delay_s']:.1f}s")
        for _ in range(int(outro_s*fps)):
            writer.append_data(cv2.cvtColor(end, cv2.COLOR_BGR2RGB))
    size = path.stat().st_size/1e6
    if not quiet:
        print(f'Video saved: {path} ({duration:.1f} sec, {size:.1f} MB)')
    return str(path), round(duration, 1)
