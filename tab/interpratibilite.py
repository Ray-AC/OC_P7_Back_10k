from imports import *

dataframe_for_lime = pd.read_csv("./data/dataframe_for_lime_10k_rows.csv")
dataframe_for_dic_for_lime = pd.read_csv("./data/dataframe_for_dic_for_lime_10k_rows.csv")

best_lgb = joblib.load('D:/Downloads/best_lightgbm_model.pkl')

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

async def tab_interpratibilite(sk_id_curr_value: int):
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