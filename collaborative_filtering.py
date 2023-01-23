# Tal Ishon
import time
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


class Recommender:
    def __init__(self, strategy='user'):
        self.pred = None
        self.user_item_matrix = None
        self.strategy = strategy
        self.similarity = np.NaN

    def fit(self, matrix):
        self.user_item_matrix = matrix
        self.similarity = pairwise_distances

        # if user - each data frame's row represents a vector of a user
        if self.strategy == 'user':
            # User - User based collaborative filtering
            start_time = time.time()
            # calculate mean
            mean_user_rating = np.nanmean(self.user_item_matrix, axis=1).reshape(-1, 1)
            ratings_diff = (self.user_item_matrix - mean_user_rating + 0.001)
            # replace nan with 0
            ratings_diff[np.isnan(ratings_diff)] = 0
            # calculate similarity according to cosine function
            user_similarity = 1 - pairwise_distances(ratings_diff, metric='cosine')

            # calculate user prediction matrix
            user_pred = mean_user_rating + user_similarity.dot(ratings_diff) / np.array(
                [np.abs(user_similarity).sum(axis=1)]).T

            # self.pred should contain your prediction matrix. transform the matrix to dataframe
            self.pred = pd.DataFrame(user_pred.round(2), index=self.user_item_matrix.index,
                                     columns=self.user_item_matrix.columns)

            time_taken = time.time() - start_time
            print('User Model in {} seconds'.format(time_taken))

            return self

        # if item - each data frame's column represents a vector of an item
        elif self.strategy == 'item':
            # Item - Item based collaborative filtering
            start_time = time.time()
            # calculate mean
            mean_user_rating = np.nanmean(self.user_item_matrix, axis=1).reshape(-1, 1)
            ratings_diff = (self.user_item_matrix - mean_user_rating + 0.001)
            # replace nan with 0
            ratings_diff[np.isnan(ratings_diff)] = 0
            # calculate similarity according to cosine function
            item_similarity = 1 - pairwise_distances(ratings_diff.T, metric='cosine')

            # calculate item prediction matrix
            item_pred = (mean_user_rating + ratings_diff.dot(item_similarity) / np.array(
                [np.abs(item_similarity).sum(axis=1)])).to_numpy()

            # self.pred should contain your prediction matrix. transform the matrix to dataframe
            self.pred = pd.DataFrame(item_pred.round(2), index=self.user_item_matrix.index,
                                     columns=self.user_item_matrix.columns)

            time_taken = time.time() - start_time
            print('Item Model in {} seconds'.format(time_taken))

            return self

    def recommend_items(self, user_id, k=5):
        # check if user_id not in data
        if not self.is_valid_userId(user_id):
            return None

        if self.strategy == 'user':
            return self.get_best_k(user_id, k)

        elif self.strategy == 'item':
            return self.get_best_k(user_id, k)

    def get_best_k(self, user_id, k=5):
        pred = self.pred.T[f'{user_id}']  # get only user_id data from pred
        matrix = self.user_item_matrix.T[f'{user_id}']  # get only user_id data from origin matrix

        # take only pred values that used to be nan before prediction
        pred = pred[np.isnan(matrix)]
        sort_products = np.argsort(-pred, kind='mergesort')  # add "-" for descending argsort
        k_best_products = sort_products[:k]  # get best k

        return list(pred[k_best_products].keys())

    def is_valid_userId(self, user_id):
        # check if user id is in matrix index
        return user_id in self.user_item_matrix.index
