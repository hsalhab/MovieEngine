import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

occupations = [
    "administrator",
    "artist",
    "educator",
    "engineer",
    "executive",
    "librarian",
    "marketing",
    "programmer",
    "scientist",
    "student",
    "technician",
    "writer",
    "other"]


def Toccupations():
    r_cols = ["user_id", "age", "gender", "occupation", "zip_code"]
    user_csv = pd.read_csv('ml-100k/u.user', sep='|', names=r_cols, encoding='latin-1')
    user = np.array(user_csv)

    fig, ax = plt.subplots()
    z = user[:,3]
    print(z.shape)
    height = np.zeros(13)
    for occ in z:
        try:
            height[occupations.index(occ)] += 1
        except:
            height[-1] += 1
    x = np.arange(13)
    for i in range(len(height)):
        if height[i] < 20:
            print(i)
    plt.bar(x, height=height, align = 'center') 
    plt.xticks(x, occupations, rotation=75, fontsize = 10)
    plt.show()

def fav_genre_by_occupation():
    '''
    x - age
    y - stacked bar chart of number of ratings over 4 by genre
    '''
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    data_csv = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1')
    bridge = np.array(data_csv)

    r_cols = ["user_id", "age", "gender", "occupation", "zip_code"]
    user_csv = pd.read_csv('ml-100k/u.user', sep='|', names=r_cols, encoding='latin-1')
    user = np.array(user_csv)

    r_cols = ["movieId" , "movieTitle" , "releaseDate" , "videoReleaseDate", "IMDbURL", "unknown" , "Action" , "Adventure" , "Animation" , "Childrens" , "Comedy" , "Crime" , "Documentary" , "Drama" , "Fantasy" , "FilmNoir" , "Horror" , "Musical" , "Mystery" , "Romance" , "SciFi" ,"Thriller" , "War" , "Western"]
    movies_csv = pd.read_csv('ml-100k/u.item', sep='|', names=r_cols,encoding='latin-1')
    movies = np.array(movies_csv)

    # ["user_id", "age", "gender", "occupation", "zip_code",'movie_id', 'rating', 'unix_timestamp', 
    # "movieTitle" , "releaseDate" , "videoReleaseDate", "IMDbURL", "unknown" , "Action" , "Adventure" , 
    # "Animation" , "Childrens" , "Comedy" , "Crime" , "Documentary" , "Drama" , "Fantasy" , "FilmNoir" , 
    # "Horror" , "Musical" , "Mystery" , "Romance" , "SciFi" ,"Thriller" , "War" , "Western"]
    total = [0 for i in range(100000)]
    for i in range( len(bridge) ):
        total[i] = np.append( user[bridge[i][0]-1], bridge[i][1:] )
        total[i] = np.append( total[i], movies[bridge[i][1]-1][1:] )
    
    all_genres = ["Action" , "Adventure"  , "Childrens" , "Comedy" , "Crime" , "Drama", 
    "Musical" , "Mystery" , "Romance" , "SciFi"]
    data = {}
    for occ in occupations:
        data[occ] = np.zeros(15)
    for t in total:
        if t[6] >= 4:
            try:
                occ = t[3]
                genres = t[-18:-3]
                idx = [i for i, x in enumerate(genres) if x == 1]
                for i in idx:
                    data[occ][i] += 1
            except:
                pass
    data = np.array([x for x in data.values()])

    data = np.delete(data, [2,6,8,9,10,11,12] ,1)

    percent_by_occ = [0 for i in range(len(occupations))]
    for i in range(len(occupations)):
        percent_by_occ[i] = [x / sum(data[i,:]) for ind, x in enumerate(data[i,:])]
    percent_by_occ = np.array(percent_by_occ)
    b = np.zeros(len(occupations))
    for i in range(8):
        d = [ x[i] for x in percent_by_occ ]
        plt.bar(range(len(occupations)), d , bottom=b)
        b = b + d
    plt.xticks(range(len(occupations)), occupations,rotation=75, fontsize = 10)
    plt.ylabel('Movie Reviews')
    plt.xlabel('Occupation')
    plt.title('Distribution of Movie Reviews Above 4/5 By Occupation')

    plt.legend(all_genres,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def movie_genre():
    r_cols = ["movieId" , "movieTitle" , "releaseDate" , "videoReleaseDate", "IMDbURL", "unknown" , "Action" , "Adventure" , "Animation" , "Childrens" , "Comedy" , "Crime" , "Documentary" , "Drama" , "Fantasy" , "FilmNoir" , "Horror" , "Musical" , "Mystery" , "Romance" , "SciFi" ,"Thriller" , "War" , "Western"]
    movies = pd.read_csv('ml-100k/u.item', sep='|', names=r_cols,encoding='latin-1')

    fig, ax = plt.subplots()
    z = np.zeros(16)
    for m in movies.itertuples():
        m = m[-18:-2]
        z += m
    x = np.arange(16)
    plt.bar(x, height= z, align = 'center') 
    plt.xticks(x, r_cols[-18:-2], rotation=75, fontsize = 10)
    plt.show()



def fav_genre_by_age():
    '''
    x - age
    y - stacked bar chart of number of ratings over 4 by genre
    '''
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    data_csv = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1')
    bridge = np.array(data_csv)

    r_cols = ["user_id", "age", "gender", "occupation", "zip_code"]
    user_csv = pd.read_csv('ml-100k/u.user', sep='|', names=r_cols, encoding='latin-1')
    user = np.array(user_csv)

    r_cols = ["movieId" , "movieTitle" , "releaseDate" , "videoReleaseDate", "IMDbURL", "unknown" , "Action" , "Adventure" , "Animation" , "Childrens" , "Comedy" , "Crime" , "Documentary" , "Drama" , "Fantasy" , "FilmNoir" , "Horror" , "Musical" , "Mystery" , "Romance" , "SciFi" ,"Thriller" , "War" , "Western"]
    movies_csv = pd.read_csv('ml-100k/u.item', sep='|', names=r_cols,encoding='latin-1')
    movies = np.array(movies_csv)

    # ["user_id", "age", "gender", "occupation", "zip_code",'movie_id', 'rating', 'unix_timestamp', 
    # "movieTitle" , "releaseDate" , "videoReleaseDate", "IMDbURL", "unknown" , "Action" , "Adventure" , 
    # "Animation" , "Childrens" , "Comedy" , "Crime" , "Documentary" , "Drama" , "Fantasy" , "FilmNoir" , 
    # "Horror" , "Musical" , "Mystery" , "Romance" , "SciFi" ,"Thriller" , "War" , "Western"]
    total = [0 for i in range(100000)]
    for i in range( len(bridge) ):
        total[i] = np.append( user[bridge[i][0]-1], bridge[i][1:] )
        total[i] = np.append( total[i], movies[bridge[i][1]-1][1:] )
    
    ages = [10, 20, 30, 40, 50, 60]
    all_genres = ["Action" , "Adventure"  , "Childrens" , "Comedy" , "Crime" , "Drama", 
    "Musical" , "Mystery" , "Romance" , "SciFi"]
    data = {10:np.zeros([15]), 20:np.zeros([15]), 30:np.zeros([15]), 40:np.zeros([15]), 50:np.zeros([15]), 60:np.zeros([15])}
    for t in total:
        if t[6] >= 4:
            try:
                r_age = round( t[1]/10 )*10
                genres = t[-18:-3]
                idx = [i for i, x in enumerate(genres) if x == 1]
                for i in idx:
                    data[r_age][i] += 1
            except:
                pass
    data = np.array([x for x in data.values()])
    data = np.delete(data, [2,6,8,9,10,11,12] ,1)
    percent_by_age = [0 for i in range(6)]
    for i in range(6):
        percent_by_age[i] = [x / sum(data[i,:]) for ind, x in enumerate(data[i,:])]
    percent_by_age = np.array(percent_by_age)
    b = np.zeros(6)
    for i in range(8):
        d = [ x[i] for x in percent_by_age  ]
        plt.bar(range(6), d , bottom=b)
        b = b + d
    plt.xticks(range(6), ages)
    plt.ylabel('Movie Reviews')
    plt.xlabel('Age Bracket')
    plt.title('Distribution of Movie Reviews Above 4/5 By Age')
    plt.legend(all_genres,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def fav_genre_by_gender():
    '''
    x - age
    y - stacked bar chart of number of ratings over 4 by genre
    '''
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    data_csv = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols,encoding='latin-1')
    bridge = np.array(data_csv)

    r_cols = ["user_id", "age", "gender", "occupation", "zip_code"]
    user_csv = pd.read_csv('ml-100k/u.user', sep='|', names=r_cols, encoding='latin-1')
    user = np.array(user_csv)

    r_cols = ["movieId" , "movieTitle" , "releaseDate" , "videoReleaseDate", "IMDbURL", "unknown" , "Action" , "Adventure" , "Animation" , "Childrens" , "Comedy" , "Crime" , "Documentary" , "Drama" , "Fantasy" , "FilmNoir" , "Horror" , "Musical" , "Mystery" , "Romance" , "SciFi" ,"Thriller" , "War" , "Western"]
    movies_csv = pd.read_csv('ml-100k/u.item', sep='|', names=r_cols,encoding='latin-1')
    movies = np.array(movies_csv)

    # ["user_id", "age", "gender", "occupation", "zip_code",'movie_id', 'rating', 'unix_timestamp', 
    # "movieTitle" , "releaseDate" , "videoReleaseDate", "IMDbURL", "unknown" , "Action" , "Adventure" , 
    # "Animation" , "Childrens" , "Comedy" , "Crime" , "Documentary" , "Drama" , "Fantasy" , "FilmNoir" , 
    # "Horror" , "Musical" , "Mystery" , "Romance" , "SciFi" ,"Thriller" , "War" , "Western"]
    total = [0 for i in range(100000)]
    for i in range( len(bridge) ):
        total[i] = np.append( user[bridge[i][0]-1], bridge[i][1:] )
        total[i] = np.append( total[i], movies[bridge[i][1]-1][1:] )
    
    all_genres = ["Action" , "Adventure"  , "Childrens" , "Comedy" , "Crime" , "Drama", 
    "Musical" , "Mystery" , "Romance" , "SciFi"]
    data = {}
    gender = ["M", "F"]
    for occ in gender:
        data[occ] = np.zeros(15)
    for t in total:
        if t[6] >= 4:
            try:
                occ = t[2]
                genres = t[-18:-3]
                idx = [i for i, x in enumerate(genres) if x == 1]
                for i in idx:
                    data[occ][i] += 1
            except:
                pass
    data = np.array([x for x in data.values()])

    data = np.delete(data, [2,6,8,9,10,11,12] ,1)

    percent_by_occ = [0 for i in range(len(gender))]
    for i in range(len(gender)):
        percent_by_occ[i] = [x / sum(data[i,:]) for ind, x in enumerate(data[i,:])]
    percent_by_occ = np.array(percent_by_occ)
    b = np.zeros(len(gender))
    for i in range(8):
        d = [ x[i] for x in percent_by_occ ]
        plt.bar(range(len(gender)), d , bottom=b)
        b = b + d
    plt.xticks(range(len(gender)), ["Male", "Female"], fontsize = 10)
    plt.ylabel('Movie Reviews')
    plt.xlabel('Gender')
    plt.title('Distribution of Movie Reviews Above 4/5 By Gender')

    plt.legend(all_genres,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


# Toccupations()
# fav_genre_by_occupation()
# fav_genre_by_age()
fav_genre_by_gender()