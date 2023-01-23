# Tal Ishon

def watch_data_info(data):

    # This function returns the first 5 rows for the object based on position.
    # It is useful for quickly testing if your object has the right type of data in it.
    print(data.head())

    # This method prints information about a DataFrame including the index dtype and column dtypes, non-null values and memory usage.
    print(data.info())

    # Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
    print(data.describe(include='all').transpose())


def print_data(data):
    # find unique appearance of each userID
    print(f"number of users are :  {len(data['UserId'].unique())}")
    # find unique appearance of each ProductID
    print(f"number of products ranked are : {len(data['ProductId'].unique())}")
    # find number of rating which is equivalent to number of ranking
    print(f"number of ranking are: {len(data['Rating'])}")
    # find min and max number of ranking to a product
    product_counts = data['ProductId'].value_counts()
    print(f"minimum number of ratings given to a product : {product_counts.loc[product_counts.idxmin()]}")
    print(f"maximum number of ratings given to a product : {product_counts.loc[product_counts.idxmax()]}")
    # find min and max number of ranking by a user
    user_counts = data['UserId'].value_counts()
    print(f"minimum number of products ratings by user : {user_counts.loc[user_counts.idxmin()]}")
    print(f"maximum number of products ratings by user : {user_counts.loc[user_counts.idxmax()]}")



