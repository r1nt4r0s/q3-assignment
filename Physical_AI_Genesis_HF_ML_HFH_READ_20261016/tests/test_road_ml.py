import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from road_ml import RISK_BY_LABEL, label_risk, expected_policy_risk, select_station, safe_speed_limit

def test_7_categories_and_mapping():
    assert len(RISK_BY_LABEL) == 7
    cls = list(RISK_BY_LABEL)
    p = np.full(len(cls), 1/len(cls))
    risk = expected_policy_risk(p, cls)
    assert .0 <= risk <= 1.
    assert np.isclose(expected_policy_risk(np.eye(len(cls))[0], cls), label_risk(cls[0]))

def test_choice_and_safe_speed():
    loc = {'A':(10., 0.),'B':(1., 0.)}
    r = {'A':.95,'B':.2}
    assert select_station((0, 0), loc, r, policy = 'nearest') == 'B'
    assert select_station((0, 0), loc, r, policy = 'adaptive') == 'B' # distance outweighs urgency
    assert select_station((0, 0), {'A':(3., 0.),'B':(1., 0.)}, r, policy = 'adaptive') == 'A'
    assert safe_speed_limit(.95)<safe_speed_limit(.10)
    assert safe_speed_limit(.98, nominal_mps = .1) <= .1
