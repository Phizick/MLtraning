import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy, Prediction
from surprise.model_selection import cross_validate
import difflib
import random

ratings_data = pd.read_csv('./data/ratings.csv')
books_metadata = pd.read_csv('./data/books.csv')

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_data[['user_id', 'book_id', 'rating']], reader)

svd = SVD(verbose=True, n_epochs=10)

results = cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
print("RMSE: ", np.mean(results['test_rmse']))
print("MAE: ", np.mean(results['test_mae']))

trainset = data.build_full_trainset()
svd.fit(trainset)

prediction = svd.predict(uid=15, iid=85)
print("prediction:", prediction)


def get_book_id(book_title, metadata):
    existing_titles = list(metadata['title'].values)
    closest_titles = difflib.get_close_matches(book_title, existing_titles)
    book_id = metadata[metadata['title'] == closest_titles[0]]['id'].values[0]
    return book_id


def get_book_info(book_id, metadata):
    book_info = metadata[metadata['id'] == book_id][['id',
                                                     'authors', 'title', 'original_title', 'original_publication_year']]
    return book_info.to_dict(orient='records')


def predict_review(user_id, book_title, model, metadata):
    book_id = get_book_id(book_title, metadata)
    review_prediction = model.predict(uid=user_id, iid=book_id)
    return review_prediction.est


def generate_recommendation(user_id, model, metadata, thresh=4):
    book_titles = list(metadata['title'].values)
    random.shuffle(book_titles)

    for book_title in book_titles:
        rating = predict_review(user_id, book_title, model, metadata)
        if rating >= thresh:
            book_id = get_book_id(book_title, metadata)
            return get_book_info(book_id, metadata)


final = generate_recommendation(10, svd, books_metadata)
print('Рекомендация:', final)

