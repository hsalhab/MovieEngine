import numpy as np
import pandas as pd
import engine

class knn_model:

    ratings = None
    similarities = None

    def __init__(self, rating, similarity):
        self.ratings = rating
        self.similarities = similarity

    def predict(self, k):
        '''
        Used the K-Top No Bias Approach with only user type
        '''
        pred = np.zeros(self.ratings.shape)

        user_bias = self.ratings.mean(axis=1)
        ratings = (self.ratings - user_bias[:, np.newaxis]).copy()

        for i in range(self.ratings.shape[0]):
            top_k_users = [np.argsort(self.similarities[:,i])[:-k-1:-1]]
            for j in range(self.ratings.shape[1]):
                pred[i, j] = self.similarities[i, :][top_k_users].dot(self.ratings[:, j][top_k_users])
                pred[i, j] /= np.sum(np.abs(self.similarities[i, :][top_k_users]))
        pred += user_bias[:, np.newaxis]

        return pred


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


if __name__ == "__main__":
    train, test = get_data()

    user_similarity = engine.fast_similarity(train, kind='user')

    knn = knn_model(train, user_similarity)
    k = 50
    pred = knn.predict(k)
    print('User-based KNN MSE:', str(engine.get_mse(pred, test)), 'at K =', str(k))
