from imports import *

async def tab_get_unique_client_ids(predict_df):
    unique_client_ids = predict_df['sk-id-curr'].unique().tolist()
    return unique_client_ids