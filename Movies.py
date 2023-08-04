
# Import des libraires
import pandas as pd
import numpy as np
import streamlit as st
from numpy import sqrt
import sqlite3


movies_df = pd.read_csv("movies.csv")
ratings_df = pd.read_csv("ratings.csv")

def get_movie_recommendations(userInput, movies_df, ratings_df, top_n=10):
    # Récupérer les utilisateurs ayant vu les films notés par notre utilisateur actif
    inputMovies = pd.DataFrame(userInput)
    inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
    inputMovies = pd.merge(inputId, inputMovies)
    userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]

    # Regroupons ensuite les lignes par userID
    userSubsetGroup = userSubset.groupby(['userId'])
    userSubsetGroup = sorted(userSubsetGroup, key=lambda x: len(x[1]), reverse=True)

    # Constituer un sous-ensemble de 100 utilisateurs les plus similaires
    userSubsetGroup = userSubsetGroup[0:100]

    pearsonCorrelationDict = {}
    for name, group in userSubsetGroup:
        group = group.sort_values(by='movieId')
        inputMovies = inputMovies.sort_values(by='movieId')
        nRatings = len(group)

        temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
        tempRatingList = temp_df['rating'].tolist()
        tempGroupList = group['rating'].tolist()

        Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList), 2) / float(nRatings)
        Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList), 2) / float(nRatings)
        Sxy = sum(i * j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList) * sum(tempGroupList) / float(nRatings)

        if Sxx != 0 and Syy != 0:
            pearsonCorrelationDict[name] = Sxy / sqrt(Sxx * Syy)
        else:
            pearsonCorrelationDict[name] = 0

    pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
    pearsonDF.columns = ['similarityIndex']
    pearsonDF['userId'] = pearsonDF.index
    pearsonDF.index = range(len(pearsonDF))

    # Récupérer les utilisateurs les plus similaires
    topUsers = pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]

    # Fusionner les utilisateurs similaires avec les notes des films
    topUsersRating = topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')

    # Multiplier l'index de similarité par les ratings
    topUsersRating['weightedRating'] = topUsersRating['similarityIndex'] * topUsersRating['rating']

    # Somme des colonnes correspondantes aux Top Users, après avoir groupé par movieId
    tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex', 'weightedRating']]
    tempTopUsersRating.columns = ['sum_similarityIndex', 'sum_weightedRating']

    # Créer un dataframe vide
    recommendation_df = pd.DataFrame()

    # Calculer la moyenne pondérée
    recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating'] / tempTopUsersRating['sum_similarityIndex']

    # Ordonner les films par score de recommandation
    recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)

    # Récupérer les détails des films recommandés
    recommendation_dfinal = recommendation_df.merge(movies_df, on='movieId')

    # Obtenir les noms des films recommandés
    recommended_movies = movies_df.loc[movies_df['movieId'].isin(recommendation_dfinal.head(top_n)['movieId'].tolist())]

    return recommended_movies

# Exemple d'utilisation
userInput = [
    {'title': 'Breakfast Club, The', 'rating': 5},
    {'title': 'Toy Story', 'rating': 3.5},
    {'title': 'Jumanji', 'rating': 2},
    {'title': 'Pulp Fiction', 'rating': 5},
    {'title': 'Akira', 'rating': 4.5}
]

movies_df = pd.read_csv("movies.csv")
ratings_df = pd.read_csv("ratings.csv")


recommended_movies = get_movie_recommendations(userInput, movies_df, ratings_df, top_n=10)
print(recommended_movies)


def movie_recommendation_app():
    # Charger les données
    movies_df = pd.read_csv("movies.csv")
    ratings_df = pd.read_csv("ratings.csv")

    # Titre de l'application
    st.title("Système de Recommandation de Films")

    # Formulaire pour que l'utilisateur entre ses notes
    st.subheader('Entrez vos notes pour les films:')
    userInput = []
    for i in range(5):
        title = st.text_input(f"Titre du film {i+1}", '')
        rating = st.slider(f"Note pour le film {i+1}", min_value=0.5, max_value=5.0, step=0.5)
        if title and rating:
            userInput.append({'title': title, 'rating': rating})

    # Bouton pour lancer la recommandation
    if st.button('Obtenir les recommandations'):
        if userInput:
            recommended_movies = get_movie_recommendations(userInput, movies_df, ratings_df, top_n=10)
            st.subheader('Films Recommandés:')
            st.dataframe(recommended_movies[['title', 'genres']])
        else:
            st.warning('Veuillez entrer au moins une note pour obtenir des recommandations.')

if __name__ == "__main__":
    movie_recommendation_app()