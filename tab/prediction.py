from imports import *

async def tab_prediction_client(predict_df, client_id: int):
    # Vérifier si le client_id est présent dans predict_df['sk-id-curr']
    if client_id not in predict_df['sk-id-curr'].values:
        raise HTTPException(status_code=404, detail="Client ID not found")
    # Obtenir la ligne correspondante du DataFrame predict_df
    client_row = predict_df[predict_df['sk-id-curr'] == client_id]
    # Convertir la ligne en dictionnaire pour le retour
    client_data = client_row.to_dict(orient='records')[0]
    return client_data