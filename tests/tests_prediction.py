import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ..tab.prediction import tab_prediction_client
from ..tab.prediction_live import tab_prediction_client_live

def test_prediction_client():
    assert tab_prediction_client(100002) == 1.0
    assert tab_prediction_client_live(100246) == 1.0