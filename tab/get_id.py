from imports import *

predict_df = pd.read_csv("./data/predict_10k_rows.csv")

async def tab_get_unique_client_ids():
    unique_client_ids = predict_df['sk-id-curr'].unique().tolist()
    return unique_client_ids