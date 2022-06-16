#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 19:08:04 2022

@author: abhishek
"""

import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
movies = pd.read_csv("/Users/abhishek/tmdb_5000_movies.csv")
credits = pd.read_csv("/Users/abhishek/tmdb_5000_credits.csv")
print(movies.head())
print(credits.head())
print(credits.head(1).values)
movies = movies.merge(credits,on='title')
print(movies.shape)
print(movies.head())
print(movies.info())
movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]
print(movies.head())
print(movies.isnull().sum())
movies = movies.dropna()
print(movies.isnull().sum())
print(movies.duplicated().sum())

def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
movies['genres']=movies['genres'].apply(convert)
movies['keywords']=movies['keywords'].apply(convert)
print(movies.head())
print(movies['genres'].head())
print(movies['keywords'].head())


def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L
movies['cast']=movies['cast'].apply(convert3)
print(movies['cast'].head())


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L
    
movies['crew']=movies['crew'].apply(fetch_director)
print(movies['crew'].head())

print(movies.head())

movies['overview']=movies['overview'].apply(lambda x:x.split())


movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])

movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']
print(movies['tags'])

new_df=movies[['movie_id','title','tags']]
new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))
print(new_df.head())

print(new_df['tags'][0])
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())
print(new_df.head())


cv=CountVectorizer(max_features=5000,stop_words='english')
vectors=cv.fit_transform(new_df['tags']).toarray()
print(vectors)
print(cv.get_feature_names())

ps=PorterStemmer()

def stem(text):
    y=[]
    
    for i in text.split():
     ps.stem(i)
    return " ".join(y) 
     
print(ps.stem('loved'))

new_df['tags']=new_df['tags'].apply(stem)


similarity=cosine_similarity(vectors)
print(similarity)


print(sorted(list(enumerate(similarity[0])),reverse=True,key=(lambda x:x[1]))[1:6])


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances=similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    for i in movies_list:
      print(new_df.iloc[i[0]].title)              
      
print(recommend('Batman Begins'))
pickle.dump(new_df,open('movies.pkl','wb'))
pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))

