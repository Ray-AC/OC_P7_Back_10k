import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from tab.prediction import tab_prediction_client
from tab.prediction_live import tab_prediction_client_live

@pytest.mark.asyncio
async def test_prediction_client():
    test_pred = await tab_prediction_client(100002)
    assert test_pred['oof_preds'] == 1.0
    test_pred_live = await tab_prediction_client_live(100246)
    assert test_pred_live['Pourcentage de chance de remboursement'] == 0.59