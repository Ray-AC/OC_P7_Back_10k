import io
import os
import lime
import pickle #
import base64
import joblib
import warnings #
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import lime.lime_tabular
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split