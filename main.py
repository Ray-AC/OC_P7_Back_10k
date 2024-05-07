from imports import *

from tab.get_id import tab_get_unique_client_ids
from tab.prediction import tab_prediction_client
from tab.prediction_live import tab_prediction_client_live
from tab.stats import tab_summary_stats_plot
from tab.data_drift import tab_data_drift
from tab.interpratibilite import tab_interpratibilite


app = FastAPI() #check query parameters

predict_df = pd.read_csv("./data/predict_10k_rows.csv")
final_dataframe = pd.read_csv("./data/final_dataframe_10k_rows.csv")
best_lgb = joblib.load('./data/best_lightgbm_model.pkl')

@app.get("/")
async def root():
    return "Vérification d'enregistrement"

@app.get("/get_unique_client_ids")
async def get_unique_client_ids():
    unique_client_ids = await tab_get_unique_client_ids(predict_df)
    return {"client_ids": unique_client_ids}

@app.get("/prediction_client")
async def prediction_client(client_id: int):
    client_data = await tab_prediction_client(predict_df, client_id)
    return client_data

@app.get("/prediction_client_live")
async def prediction_client_live(client_id: int):
    combined_dict = await tab_prediction_client_live(predict_df, final_dataframe, best_lgb, client_id)
    return combined_dict

@app.get("/summary_stats_plot")
async def summary_stats_plot(sk_id_to_display: int):
    image_base64 = await tab_summary_stats_plot(final_dataframe, sk_id_to_display)
    return image_base64

@app.get("/data_drift")
async def data_drift():
    image_base64 = await tab_data_drift()
    return image_base64

@app.get("/interpratibilite")
async def interpratibilite(sk_id_curr_value: int):
    explanation_base64 = await tab_interpratibilite(sk_id_curr_value)
    return explanation_base64