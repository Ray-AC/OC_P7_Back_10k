from imports import *

predict_df = pd.read_csv("./data/predict_10k_rows.csv")
final_dataframe = pd.read_csv("./data/final_dataframe_10k_rows.csv")

best_lgb = joblib.load('D:/Downloads/best_lightgbm_model.pkl')

async def tab_prediction_client_live(client_id: int):
    # Vérifier si le client_id est présent dans predict_df['sk-id-curr']
    if client_id not in predict_df['sk-id-curr'].values:
        raise HTTPException(status_code=404, detail="Client ID not found")
    # Obtenir les données du client correspondant du DataFrame predict_df
    client_data = final_dataframe[final_dataframe['sk-id-curr'] == client_id].drop(columns=['target', 'sk-id-curr', 'index'])
    # Effectuer la prédiction et la proba de la prediction en direct
    prediction = best_lgb.predict(client_data).tolist()[0]
    prediction_proba = best_lgb.predict_proba(client_data).tolist()[0]
    rounded_second_prediction_proba = np.round(best_lgb.predict_proba(client_data)[:, 1], 2).tolist()[0] # prediction proba arrondie au centième avec numpy.round() de la deuxieme colonne (1.0)
    combined_dict = {'prediction': prediction, 'Pourcentage de chance de remboursement': rounded_second_prediction_proba}
    return combined_dict