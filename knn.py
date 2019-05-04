import numpy as np
import pandas as pd
import engine

class knn_model:

    ratings = None
    similarities = None

    def __init__(self, rating, similarity):
        self.ratings = rating

    def predict(self, k, similarities, kind):
        '''
        Used the K-Top No Bias Approach with only user type
        '''
        pred = np.zeros(self.ratings.shape)

        if kind == 'user':
            user_bias = self.ratings.mean(axis=1)
            ratings = (self.ratings - user_bias[:, np.newaxis]).copy()

            for i in range(self.ratings.shape[0]):
                top_k_users = [np.argsort(similarities[:,i])[:-k-1:-1]]
                for j in range(self.ratings.shape[1]):
                    pred[i, j] = similarities[i, :][top_k_users].dot(self.ratings[:, j][top_k_users])
                    pred[i, j] /= np.sum(np.abs(similarities[i, :][top_k_users]))
            pred += user_bias[:, np.newaxis]

        if kind == 'movie':
            item_bias = self.ratings.mean(axis=0)
            ratings = (self.ratings - item_bias[np.newaxis, :]).copy()

            for j in range(self.ratings.shape[1]):
                top_k_items = [np.argsort(similarities[:,j])[:-k-1:-1]]
                for i in range(self.ratings.shape[0]):
                    pred[i, j] = similarities[j, :][top_k_items].dot(self.ratings[i, :][top_k_items].T)
                    pred[i, j] /= np.sum(np.abs(similarities[j, :][top_k_items]))
            pred += item_bias[np.newaxis, :]

        return pred

from tqdm import tqdm

def get_data():
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv('ml-100k/u.data', sep='\t', names=names)

    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    ratings = np.zeros((n_users, n_items))
    for row in df.itertuples():
        ratings[row[1]-1, row[2]-1] = row[3]

    train, test = engine.train_test_split(ratings)

    return train, test

def getRecommendations(movie_similarity):
    m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
    df = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, encoding = "latin-1")

    idx_to_movie = {}
    for row in df.itertuples():
        idx_to_movie[row.Index[0]] = row.Index[1]

    def top_k_movies(similarity, mapper, movie_idx, k=6):
        print(similarity[movie_idx,:])
        max, min = (0,0), (1000,0)
        for i,s in enumerate(similarity[movie_idx,:]):
            if s > max[0]:
                max = (s,i)
            elif s < min[0]:
                min = (s,i)
        print("min:", min)
        print("max:", max)
        print(np.argsort(similarity[movie_idx,:]))
        print(np.argsort(similarity[movie_idx,:])[:-k-1:-1])
        return [mapper[x] for x in np.argsort(similarity[movie_idx,:])[:-k-1:-1]]

    idx = 89 # Toy Story
    movies = top_k_movies(movie_similarity, idx_to_movie, idx)
    print(idx_to_movie[idx])
    print(movies)
    # print([idx_to_movie[x] for x in movies])

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

if __name__ == "__main__":
    train, test = get_data()

    # user_similarity = engine.fast_similarity(train, kind='user')
    # movie_similarity = engine.fast_similarity(train, kind='item')

    from sklearn.metrics import pairwise_distances
    # # Convert from distance to similarity
    movie_similarity = 1 - pairwise_distances(train.T, metric='correlation')
    movie_similarity[np.isnan(movie_similarity)] = 0.

    m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
    df = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, encoding = "latin-1")

    idx_to_movie = {}
    for row in df.itertuples():
        idx_to_movie[row.Index[0]-1] = row.Index[1]
    
    tsne = TSNE( perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    plot_only = 500
    low_dim_embs = tsne.fit_transform(movie_similarity[:plot_only, :])
    labels = [idx_to_movie[i] for i in range(plot_only)]
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels[:100]):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label.split("(")[0], xy=(x, y), xytext=(0, 0), textcoords='offset points',fontsize='x-small', ha='center', va='bottom')
    # plt.savefig("./tsne.png"
    plt.show()

    from sklearn.metrics import pairwise_distances
    # Convert from distance to similarity
    item_correlation = 1 - pairwise_distances(train.T, metric='correlation')
    item_correlation[np.isnan(item_correlation)] = 0.

    knn = knn_model(train, movie_similarity)
    k = 50
    # user_pred = knn.predict(k, user_similarity, 'user')
    movie_pred = knn.predict(k, movie_similarity,  'movie')
    print('User-based KNN MSE:', str(engine.get_mse(movie_pred, test)), 'at K =', str(k))

    getRecommendations(movie_similarity)
