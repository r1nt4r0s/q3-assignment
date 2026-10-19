"""Simple risk-dependent speed *surrogate* training environment for PPO.
It is NOT Genesis physics. Policies are evaluated again in Genesis before claiming
physical-task performance.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SpeedEnv(gym.Env):
    metadata = {'render_modes':[]}
    def __init__(self, seed = None):
        super().__init__()
        self.observation_space = spaces.Box(0., 1., shape = (4, ), dtype = np.float32)
        self.action_space = spaces.Box(0., 1., shape = (1, ), dtype = np.float32)
        self.rng = np.random.default_rng(seed)
        self.distance = 18.
        self.dt = .4
        self.maxspeed = .65
        self.horizon = 210
        self.progress = 0.
        self.v = 0.
        self.steps = 0
        self.zones = []
    def _risk(self):
        return float(np.clip(sum(v*np.exp(-.5*((self.progress-c)/w)**2) for c, v, w in self.zones), 0, 1))
    def _obs(self):
        return np.asarray([self._risk(), self.v/self.maxspeed, min(1, self.progress/self.distance), self.steps/self.horizon], dtype = np.float32)
    def reset(self, *, seed = None, options = None):
        super().reset(seed = seed)
        self.rng = np.random.default_rng(seed) if seed is not None else self.rng
        self.progress = 0.
        self.v = 0.
        self.steps = 0
        self.zones = [(float(self.rng.uniform(3, 15)), float(self.rng.uniform(.4, .98)), float(self.rng.uniform(.6, 1.8))) for _ in range(3)]
        return self._obs(), {}
    def step(self, action):
        act = float(np.clip(np.asarray(action).reshape(-1)[0], 0, 1))
        prev = self.v
        # Acceleration-limited first-order response (surrogate dynamics)
        self.v = float(np.clip(self.v+np.clip(self.maxspeed*act-self.v, -.12, .12), 0, self.maxspeed))
        risk = self._risk()
        overspeed = max(0., self.v/self.maxspeed-(.25+.50*(1.-risk)))
        dx = self.v*self.dt
        self.progress += dx
        self.steps += 1
        completed = self.progress >= self.distance
        truncated = self.steps >= self.horizon
        reward = 4.*dx-.025 - 1.6*risk*overspeed**2-.15*abs(self.v-prev)
        if completed:
            reward += 5.
        return self._obs(), float(reward), bool(completed), bool(truncated), {'risk':risk,'mps':self.v,'dx':dx,'overspeed':overspeed}


def eval_speed_policy(policy, seeds, ppo = None):
    from road_ml import safe_speed_limit
    results = []
    for seed in seeds:
        env = SpeedEnv()
        obs, _ = env.reset(seed = int(seed))
        done = False
        traces = []
        while not done:
            risk = float(obs[0])
            if policy == 'constant':
                a = np.array([1.], dtype = np.float32)
            elif policy == 'rule':
                a = np.array([safe_speed_limit(risk, env.maxspeed)/env.maxspeed], dtype = np.float32)
            elif policy == 'ppo' and ppo is not None:
                a, _ = ppo.predict(obs, deterministic = True)
            else:
                raise ValueError(policy)
            obs, reward, terminated, truncated, info = env.step(a)
            traces.append(info)
            done = terminated or truncated
        results.append({'seed':seed,'policy':policy,'finished':env.progress>=env.distance,
                        'travel_time_s':env.steps*env.dt,'sum_risk_overspeed_sq':sum(t['risk']*t['overspeed']**2 for t in traces),
                        'mean_speed_mps':float(np.mean([t['mps'] for t in traces]))})
    return results
