"""Road issue classification utilities for the PAI final project.

The Hugging Face road issues data has *issue category* labels, not defect
severity or normal-vs-abnormal labels. The curated risk score below is a
policy assumption (not learned severity, not a calibrated probability).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

REPO_ID = 'Programmer-RD-AI/road-issues-detection-dataset'
RISK_BY_LABEL = {
    'Pothole Issues': 0.95,
    'Damaged Road issues': 0.75,
    'Broken Road Sign Issues': 0.70,
    'Illegal Parking Issues': 0.55,
    'Mixed Issues': 0.85,
    'Littering Garbage on Public Places Issues': 0.40,
    'Vandalism Issues': 0.35,
}


def label_risk(label: str) -> float:
    if label not in RISK_BY_LABEL:
        raise ValueError(f'Unknown dataset label: {label!r}. Revisit the risk policy mapping.')
    return float(RISK_BY_LABEL[label])


def expected_policy_risk(class_probabilities, class_names):
    """Expected *heuristic* risk: not probability of a severe physical hazard."""
    vals = np.asarray([label_risk(str(n)) for n in class_names], dtype = float)
    p = np.asarray(class_probabilities, dtype = float)
    if p.shape[-1] != len(vals):
        raise ValueError('class dimension mismatch')
    return np.sum(p*vals, axis = -1)


def select_station(current_xy, locations, scores, visited = None, policy = 'adaptive'):
    """Distance vs learned-category-dependent policy comparison.

    Note: straight-line distance is for target priority only, not obstacle-safe
    path length. The physical controller independently uses A*.
    """
    visited = set(visited or ())
    available = [s for s in locations if s not in visited]
    if not available:
        return None
    pos = np.asarray(current_xy, dtype = float)
    dd = {k:float(np.linalg.norm(pos-np.asarray(locations[k]))) for k in available}
    if policy == 'nearest':
        return min(available, key = lambda k:(dd[k], k))
    if policy != 'adaptive':
        raise ValueError('policy should be nearest or adaptive')
    return max(available, key = lambda k:(3.5*float(scores[k]) - 0.30*dd[k], -dd[k]))


def safe_speed_limit(risk, nominal_mps = 0.65):
    """Interpretable, deliberately conservative safety shield (simulation only)."""
    risk = float(np.clip(risk, 0., 1.))
    if risk >= 0.80:
        return min(nominal_mps, 0.20)
    if risk >= 0.60:
        return min(nominal_mps, 0.32)
    if risk >= 0.40:
        return min(nominal_mps, 0.48)
    return float(nominal_mps)


def save_station_manifest(items, path):
    Path(path).write_text(json.dumps(items, ensure_ascii = False, indent = 2), encoding = 'utf8')


class RoadVision:
    def __init__(self, checkpoint_path, device = 'cpu'):
        import torch
        import torchvision
        from torchvision.models import mobilenet_v3_small
        ckpt = torch.load(str(checkpoint_path), map_location = 'cpu', weights_only = True)
        self.class_names = [str(x) for x in ckpt['labels']]
        # Model weights from the training notebook; do not silently use ImageNet labels.
        self.model = mobilenet_v3_small(weights = None, num_classes = len(self.class_names))
        self.model.load_state_dict(ckpt['state_dict'])
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.preprocess=torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean = (.485, .456, .406), std = (.229, .224, .225))
        ])
        self.temperature = float(ckpt.get('temperature', 1.0))
        self.source = ckpt.get('source','unknown')

    def predict(self, img):
        import torch
        from PIL import Image
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        x = self.preprocess(img.convert('RGB')).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probs = torch.softmax(self.model(x)[0]/self.temperature, dim = 0).cpu().numpy()
        predicted = int(np.argmax(probs))
        return {
            'label':self.class_names[predicted],
            'confidence':float(probs[predicted]),
            'policy_risk':float(expected_policy_risk(probs, self.class_names)),
            'category_probabilities':{n:float(p) for n, p in zip(self.class_names, probs)},
            'risk_is_heuristic':True,
        }
