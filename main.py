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
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

app = FastAPI() #check query parameters

#Version allégée à 10k lignes pour déploiement

predict_df = pd.read_csv("./data/predict_10k_rows.csv")

final_dataframe = pd.read_csv("./data/final_dataframe_10k_rows.csv")
dataframe_for_dic_for_lime = pd.read_csv("./data/dataframe_for_dic_for_lime_10k_rows.csv")
dataframe_for_lime = pd.read_csv("./data/dataframe_for_lime_10k_rows.csv")

best_lgb = joblib.load('D:/Downloads/best_lightgbm_model.pkl')
with open("D:/Downloads/data_drift.png", "rb") as file:
    image_content = file.read()
indice_sk_id_curr = {}

# Parcourir chaque ligne du DataFrame df_usable
for index, row in dataframe_for_dic_for_lime.iterrows():
    # Récupérer la valeur de 'sk-id-curr' pour cette ligne et la convertir en entier
    sk_id_curr_value = int(row['sk-id-curr'])
    # Associer l'indice au sk-id-curr dans le dictionnaire
    indice_sk_id_curr[index] = sk_id_curr_value
    
# Création d'un explainer LIME
explainer = lime.lime_tabular.LimeTabularExplainer(dataframe_for_lime.drop(columns=['target']).values,
                                                feature_names=dataframe_for_lime.drop(columns=['target']).columns,
                                                class_names=['Non-Default', 'Default'],
                                                discretize_continuous=True)

@app.get("/")
async def root():
    return "Vérification d'enregistrement"

@app.get("/get_unique_client_ids")
async def get_unique_client_ids():
    unique_client_ids = predict_df['sk-id-curr'].unique().tolist()
    return unique_client_ids

@app.get("/prediction_client")
async def prediction_client(client_id: int):
    # Vérifier si le client_id est présent dans predict_df['sk-id-curr']
    if client_id not in predict_df['sk-id-curr'].values:
        raise HTTPException(status_code=404, detail="Client ID not found")
    # Obtenir la ligne correspondante du DataFrame predict_df
    client_row = predict_df[predict_df['sk-id-curr'] == client_id]
    # Convertir la ligne en dictionnaire pour le retour
    client_data = client_row.to_dict(orient='records')[0]
    return client_data

@app.get("/prediction_client_live")
async def prediction_client_live(client_id: int):
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

@app.get("/summary_stats_plot")
async def summary_stats_plot(sk_id_to_display: int):
    # Sélectionner les colonnes spécifiées
    final_dataframe_subset = final_dataframe.loc[:, ['sk-id-curr', 'payment-rate', 'ext-source-3', 'ext-source-2', 'ext-source-1', 'days-birth', 'amt-annuity', 'days-employed']]
    # Vérifier si le sk_id_to_display est présent dans final_dataframe_subset['sk-id-curr']
    if sk_id_to_display not in final_dataframe_subset['sk-id-curr'].values:
        raise HTTPException(status_code=404, detail="Client ID not found")
    # Sélectionner les informations du client spécifique
    selected_row = final_dataframe_subset.loc[final_dataframe_subset['sk-id-curr'] == sk_id_to_display]
    # Calculer les statistiques récapitulatives pour les autres clients
    summary_stats = final_dataframe_subset.drop(selected_row.index).describe()
    # Ajouter les informations du client spécifique au DataFrame des statistiques récapitulatives
    summary_stats.loc['Selected Client'] = selected_row.iloc[0, :]
    # Supprimer la ligne 'count' et la colonne 'sk-id-curr' du DataFrame summary_stats
    summary_stats = summary_stats.drop(index='count', columns='sk-id-curr')
    # Créer une figure et des sous-graphiques en grille
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    # Itérer sur les colonnes et créer un graphique à barres pour chaque colonne
    for i, column in enumerate(summary_stats.columns):
        row_index = i // 3  # Index de ligne dans la grille
        col_index = i % 3   # Index de colonne dans la grille
        # Créer un tableau de couleurs pour les barres
        colors = ['lightgrey'] * len(summary_stats.index)
        # Trouver l'indice du client sélectionné dans les index
        selected_index = summary_stats.index.get_loc('Selected Client')
        # Mettre la couleur du client sélectionné en rouge
        colors[selected_index] = 'red'
        # Tracer le graphique à barres dans le sous-graphique correspondant
        axs[row_index, col_index].bar(summary_stats.index, summary_stats[column], color=colors)
        axs[row_index, col_index].set_title(f'Comparison of {column} for Other Clients vs. Selected Client')
        axs[row_index, col_index].set_xlabel('Clients')
        axs[row_index, col_index].set_ylabel('Values')
        axs[row_index, col_index].grid(axis='y', linestyle='--', alpha=0.7)
        axs[row_index, col_index].tick_params(axis='x', rotation=45)
        axs[row_index, col_index].set_xticks(summary_stats.index)  # Assurez-vous que toutes les étiquettes d'axe x sont affichées
        axs[row_index, col_index].set_xlim(-0.5, len(summary_stats.index)-0.5)  # Ajuster les limites de l'axe x
        axs[row_index, col_index].tick_params(axis='x', which='both', bottom=False, top=False)  # Masquer les étiquettes de l'axe x
    # Ajuster l'espacement entre les sous-graphiques
    plt.tight_layout()
    # Créer un buffer de mémoire pour stocker le graphique
    buffer = io.BytesIO()
    # Enregistrer le graphique dans le buffer
    plt.savefig(buffer, format='png')
    # Fermer le graphique pour libérer la mémoire
    plt.close()
    # Retourner le contenu du buffer en tant que réponse HTTP
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    return image_base64

@app.get("/data_drift")
async def data_drift():
    image_base64 = base64.b64encode(image_content).decode()
    return image_base64

@app.get("/interpratibilite")
async def interpratibilite(sk_id_curr_value: int):
    # Obtenir l'indice associé à la valeur de sk-id-curr
    observation_idx = list(indice_sk_id_curr.values()).index(sk_id_curr_value)
    # Sélectionner l'observation correspondante
    observation = dataframe_for_lime.drop(columns=['target']).iloc[observation_idx].values
    true_label = dataframe_for_lime['target'].iloc[observation_idx]
    # Explication de la prédiction
    explanation = explainer.explain_instance(observation, best_lgb.predict_proba, num_features=5, top_labels=1)
    # Obtenir les étiquettes disponibles en appelant la méthode
    available_labels = explanation.available_labels()
    # Créer un DataFrame à partir de la liste d'explication pour une étiquette spécifique
    explanation_df = pd.DataFrame(explanation.as_list(label=available_labels[0]))
    # Créer le graphique à barres
    plt.figure(figsize=(10, 5))
    sns.barplot(x=explanation_df[1], y=explanation_df[0], palette=['green' if val >= 0 else 'red' for val in explanation_df[1]])
    plt.title('LIME Explanation')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    # Ajouter une légende pour les couleurs
    colors = {'Positive': 'green', 'Negative': 'red'}
    legend_labels = [plt.Rectangle((0,0),1,1, color=color) for color in colors.values()]
    plt.legend(legend_labels, colors.keys())
    # Créer un buffer de mémoire pour stocker le graphique
    buffer = io.BytesIO()
    # Enregistrer le graphique dans le buffer
    plt.savefig(buffer, format='png')
    plt.close()
    # Retourner le contenu du buffer en tant que réponse HTTP
    buffer.seek(0)
    explanation_base64 = base64.b64encode(buffer.getvalue()).decode()
    return explanation_base64