# Tal Ishon
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


def RMSE(test_set, cf):
    predictions = []  # will hold all rating predictions from cf.pred
    bench_predictions = []  # will hold all mean ratings for each user id
    mean_matrix = cf.user_item_matrix.mean(axis=1)
    for row in test_set.itertuples():
        predict = cf.pred.loc[row.UserId, row.ProductId]
        predictions.append(predict)
        b_predict = mean_matrix.loc[row.UserId]
        bench_predictions.append(b_predict)

    actuals = np.array(test_set.Rating)
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    benchmark = bench_predictions
    benchmark_mse = mean_squared_error(actuals, benchmark)
    benchmark_rmse = np.sqrt(benchmark_mse)
    print("RMSE for {}-based CF: {}".format(cf.strategy, rmse.round(5)))
    print("RMSE for the benchmark: {}".format(benchmark_rmse.round(5)))


def precision_at_k(test_set, cf, k):
    precisions = []
    benchmark_precisions = []
    mean_ratings = cf.user_item_matrix.mean()
    # return highest k values of mean_ratings
    benchmark_recommended_items = mean_ratings.nlargest(k).index.tolist()

    for user_id in test_set.UserId.unique():
        recommended_items = cf.recommend_items(user_id, k)
        # calculate items of a user_id that got rate 3 or higher and get the product id according to query
        relevant_items = list(test_set.query("UserId == @user_id and Rating >= 3").ProductId)
        # Recommended@k INTERSECTION Relevant:
        relevant_items_recommended = set(recommended_items).intersection(set(relevant_items))
        # RecommendedBench@k INTERSECTION Relevant:
        relevant_and_recommended_benchmark = set(benchmark_recommended_items).intersection(set(relevant_items))

        if relevant_items and recommended_items:  # check not to divide by 0 or to skip if numerator is empty
            # calculate precision for each user
            precisions.append(len(relevant_items_recommended)/len(recommended_items))
            benchmark_precisions.append(len(relevant_and_recommended_benchmark)/len(recommended_items))

    if precisions:  # check not to divide by 0
        test_set_precision = sum(precisions) / len(precisions)  # average off all precisions
        print("user-based CF Precision@{}: {}".format(k, round(test_set_precision, 5)))
    if benchmark_precisions:  # check not to divide by 0
        benchmark = sum(benchmark_precisions) / len(benchmark_precisions)
        print("highest-ranked Benchmark Precision@{}: {}".format(k, round(benchmark, 5)))

def recall_at_k(test_set, cf, k):
    recalls = []
    benchmark_recalls = []
    mean_ratings = cf.user_item_matrix.mean()
    # return highest k values of mean_ratings
    benchmark_recommended_items = mean_ratings.nlargest(k).index.tolist()

    for user_id in test_set.UserId.unique():
        recommended_items = cf.recommend_items(user_id, k)
        # calculate items of a user_id that got rate 3 or higher and get the product id according to query
        relevant_items = list(test_set.query("UserId == @user_id and Rating >= 3").ProductId)
        # Recommended@k INTERSECTION Relevant:
        relevant_items_recommended = set(recommended_items).intersection(set(relevant_items))
        # RecommendedBench@k INTERSECTION Relevant:
        relevant_and_recommended_benchmark = set(benchmark_recommended_items).intersection(set(relevant_items))

        if recommended_items and relevant_items:   # check not to divide by 0 or to skip if numerator is empty
            # calculate precision for each user
            recalls.append(len(relevant_items_recommended) / len(relevant_items))
            benchmark_recalls.append(len(relevant_and_recommended_benchmark) / len(relevant_items))

    if recalls:  # check not to divide by 0
        test_set_recalls = sum(recalls) / len(recalls)  # average off all recalls
        print("user-based CF Recall@{}: {}".format(k, round(test_set_recalls, 5)))
    if benchmark_recalls:  # check not to divide by 0
        benchmark = sum(benchmark_recalls) / len(benchmark_recalls)
        print("highest-ranked Benchmark Recall@{}: {}".format(k, round(benchmark, 5)))

