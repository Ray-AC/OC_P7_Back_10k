from imports import *

async def tab_summary_stats_plot(final_dataframe, sk_id_to_display: int):
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