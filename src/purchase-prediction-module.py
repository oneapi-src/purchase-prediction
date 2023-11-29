'''
   This code base is adopted from the below notebook
   https://www.kaggle.com/code/fabiendaniel/customer-segmentation/notebook
'''
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
import argparse
import time
import datetime
import logging
import warnings
import pandas as pd
import numpy as np
import nltk
from joblib import load, dump

# Ensure that the required NLTK libraries are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Declaring the path where the Models will be saved and loaded from
rfc_model = 'output/models/rfc_model.joblib'
knn_model = 'output/models/knn_model.joblib'
dtc_model = 'output/models/dtc_model.joblib'

# Hyperparamter tuning, saving & loading of Machine Learning models
def hyperparameter_tuning(algorithm, param_grid, kFold):
    "Hyper parameter tuning for the given algorithms"
    gcv = GridSearchCV(algorithm, param_grid=parameters, cv=kFold, verbose=10, n_jobs=-1)
    model = gcv.fit(X_train,Y_train)
    estimator = gcv.best_estimator_
    return estimator, model

def save_model(estimator, modelname):
    logger.info("Saving the model ...")
    try:
        dump(estimator, modelname)
    except Exception as e:
        # Exception handler, alert the user
        raise IOError("Error saving model data to disk: {}".format(str(e))) from e

def load_model(modelname):
    try:
        logger.info("Model loading ...")
        loaded_model = load(modelname)
    except Exception as e:
        raise IOError("Error loading model data from disk: {}".format(str(e))) from e
    return loaded_model

def iter_minibatches(chunksize):
    # Provide chunks one by one
    chunkstartmarker = 0
    while chunkstartmarker < len(X_train):
        chunkrows = chunkstartmarker+chunksize
        X_chunk, Y_chunk = X_train[chunkstartmarker:chunkrows], Y_train[chunkstartmarker:chunkrows]
        yield X_chunk, Y_chunk
        chunkstartmarker += chunksize

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-rtd',
                        '--raw_train_data',
                        type=str,
                        required=False,
                        default='data/data.csv',
                        help='raw data csv file, if this parameter is specified then it will only perform the data preparation part')
    parser.add_argument('-daf',
                        '--data_aug_factor',
                        type=int,
                        required=False,
                        default='0',
                        help='data augmentation/multiplication factor, requires --raw-train-data parameter')
    parser.add_argument('-ftd',
                        '--final_train_data',
                        type=str,
                        required=False,
                        default='0',
                        help='final filtered data csv file, if this parameter is specified then it will skip the data preparation part')
    parser.add_argument('-t',
                        '--tuning',
                        type=str,
                        required=False,
                        default='0',
                        help='hyper parameter tuning (0/1). Along with Hyperparamter tuning, the model is saved ')
    parser.add_argument('-alg',
                        '--algorithm',
                        type=str,
                        required=False,
                        default='knn',
                        help='scikit learn classifier algorithm to be used (knn,dtc,rfc) \
                        - knn=KNearestNeighborClassifier, dtc=DecisionTreeClassifier, rfc=RandomForestClassifier')
    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        required=False,
                        default=None,
                        help='hyper parameter tuning (0/1). Along with Hyperparamter tuning, the model is saved ')
    parser.add_argument('-inf',
                        '--inference',
                        type=str,
                        required=False,
                        default='0',
                        help='Perform Inference on the saved models.Specify the model file with path i.e knn_model or rfc_model or dtc_model')
    
    parser.add_argument('-l',
                        '--logfile',
                        type=str,
                        default="",
                        help="log file to output benchmarking results to")
    
    FLAGS = parser.parse_args()

    if FLAGS.logfile == "":
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(filename=FLAGS.logfile, level=logging.DEBUG)

    logger = logging.getLogger(__name__)
    logging.getLogger('sklearnex').setLevel(logging.WARNING)

    finaltraindata = FLAGS.final_train_data
    dataaugfactor = FLAGS.data_aug_factor
    rawtraindata = FLAGS.raw_train_data
    algorithm = FLAGS.algorithm
    inference = FLAGS.inference
    tuning = True if FLAGS.tuning == '1' else False
    batch_size = FLAGS.batch_size

    prgstime = time.time()
    
    from sklearnex import patch_sklearn  # pylint: disable=import-error
    patch_sklearn()
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score
    from sklearn import model_selection
    from sklearn import metrics
    from sklearn.model_selection import GridSearchCV
    from sklearn import neighbors
    from sklearn import linear_model
    from sklearn import tree
    from sklearn import ensemble
    from sklearn.decomposition import PCA

    warnings.filterwarnings('ignore')

    if finaltraindata == '0':
        # read the datafile
        try:
            # write some code
            # that might throw exception
            logger.info(f'Reading raw data from csv file {rawtraindata}...')
            df_initial = pd.read_csv(rawtraindata, encoding="ISO-8859-1",
                                     dtype={'CustomerID': str, 'InvoiceID': str})
        except FileNotFoundError:
            # Exception handler, alert the user
            sys.exit("ALERT:Please check the input data path.Input data not found")

        print('Dataframe dimensions:', df_initial.shape)
        # ______
        df_initial['InvoiceDate'] = pd.to_datetime(df_initial['InvoiceDate'])
        # ____________________________________________________________
        # gives some infos on columns types and number of null values
        tab_info = pd.DataFrame(df_initial.dtypes).T.rename(index={0: 'column type'})
        tab_info = pd.concat([tab_info, pd.DataFrame(df_initial.isnull().sum()).T.rename(index={0: 'null values (nb)'})])
        tab_info = pd.concat([tab_info, pd.DataFrame(df_initial.isnull().sum() / df_initial.shape[0] * 100).T.
                                   rename(index={0: 'null values (%)'})])
        df_initial.dropna(axis=0, subset=['CustomerID'], inplace=True)
        print('Dataframe dimensions after removing null values:', df_initial.shape)
        # ____________________________________________________________
        # gives some infos on columns types and number of null values
        tab_info = pd.DataFrame(df_initial.dtypes).T.rename(index={0: 'column type'})
        tab_info = pd.concat([tab_info, pd.DataFrame(df_initial.isnull().sum()).T.rename(index={0: 'null values (nb)'})])
        tab_info = pd.concat([tab_info, pd.DataFrame(df_initial.isnull().sum() / df_initial.shape[0] * 100).T.
                                   rename(index={0: 'null values (%)'})])
        logger.info('Duplicated entries: {}'.format(df_initial.duplicated().sum()))
        df_initial.drop_duplicates(inplace=True)

        temp = df_initial[['CustomerID', 'InvoiceNo', 'Country']].groupby(['CustomerID', 'InvoiceNo', 'Country']).count()
        temp = temp.reset_index(drop=False)
        countries = temp['Country'].value_counts()

        pd.DataFrame([{'products': len(df_initial['StockCode'].value_counts()),
                       'transactions': len(df_initial['InvoiceNo'].value_counts()),
                       'customers': len(df_initial['CustomerID'].value_counts()),
                       }], columns=['products', 'transactions', 'customers'], index=['quantity']
                     )
        temp = df_initial.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate'].count()
        nb_products_per_basket = temp.rename(columns={'InvoiceDate': 'Number of products'})
        nb_products_per_basket[:10].sort_values('CustomerID')

        nb_products_per_basket['order_canceled'] = nb_products_per_basket['InvoiceNo'].apply(lambda x: int('C' in x))
        # ______________________________________________________________________________________________
        n1 = nb_products_per_basket['order_canceled'].sum()
        n2 = nb_products_per_basket.shape[0]

        df_check = df_initial[df_initial['Quantity'] < 0][['CustomerID', 'Quantity',
                                                           'StockCode', 'Description', 'UnitPrice']]
        for index, col in df_check.iterrows():
            if df_initial[(df_initial['CustomerID'] == col[0]) & (df_initial['Quantity'] == -col[1]) &
                          (df_initial['Description'] == col[2])].shape[0] == 0:
                break

        df_check = df_initial[(df_initial['Quantity'] < 0) & (df_initial['Description'] != 'Discount')][
                                         ['CustomerID', 'Quantity', 'StockCode', 'Description', 'UnitPrice']]

        for index, col in df_check.iterrows():
            if df_initial[(df_initial['CustomerID'] == col[0]) & (df_initial['Quantity'] == -col[1]) &
                          (df_initial['Description'] == col[2])].shape[0] == 0:
                break

        df_cleaned = df_initial.copy(deep=True)
        df_cleaned['QuantityCanceled'] = 0

        entry_to_remove = []
        doubtfull_entry = []

        for index, col in df_initial.iterrows():
            if (col['Quantity'] > 0) or col['Description'] == 'Discount':
                continue
            df_test = df_initial[(df_initial['CustomerID'] == col['CustomerID'])
                                 & (df_initial['StockCode'] == col['StockCode'])
                                 & (df_initial['InvoiceDate'] < col['InvoiceDate'])
                                 & (df_initial['Quantity'] > 0)].copy()
            # _________________________________
            # Cancelation WITHOUT counterpart
            if df_test.shape[0] == 0:
                doubtfull_entry.append(index)
            # ________________________________
            # Cancelation WITH a counterpart
            elif df_test.shape[0] == 1:
                index_order = df_test.index[0]
                df_cleaned.loc[index_order, 'QuantityCanceled'] = -col['Quantity']
                entry_to_remove.append(index)
            # ______________________________________________________________
            # Various counterparts exist in orders: we delete the last one
            elif df_test.shape[0] > 1:
                df_test.sort_index(axis=0, ascending=False, inplace=True)
                for ind, val in df_test.iterrows():
                    if val['Quantity'] < -col['Quantity']:
                        continue
                    df_cleaned.loc[ind, 'QuantityCanceled'] = -col['Quantity']
                    entry_to_remove.append(index)
                    break

        df_cleaned.drop(entry_to_remove, axis=0, inplace=True)
        df_cleaned.drop(doubtfull_entry, axis=0, inplace=True)
        remaining_entries = df_cleaned[(df_cleaned['Quantity'] < 0) & (df_cleaned['StockCode'] != 'D')]
        list_special_codes = df_cleaned[df_cleaned['StockCode'].str.contains('^[a-zA-Z]+', regex=True)]['StockCode'].unique()
        df_cleaned['TotalPrice'] = df_cleaned['UnitPrice'] * (df_cleaned['Quantity'] - df_cleaned['QuantityCanceled'])

        # sum of purchases / user & order
        temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['TotalPrice'].sum()
        basket_price = temp.rename(columns={'TotalPrice': 'Basket Price'})
        # date of order
        df_cleaned['InvoiceDate_int'] = df_cleaned['InvoiceDate'].astype('int64')
        temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate_int'].mean()
        df_cleaned.drop('InvoiceDate_int', axis=1, inplace=True)
        basket_price.loc[:, 'InvoiceDate'] = pd.to_datetime(temp['InvoiceDate_int'])
        # selection of significant entries:
        basket_price = basket_price[basket_price['Basket Price'] > 0]

        # Purchase countdown
        price_range = [0, 50, 100, 200, 500, 1000, 5000, 50000]
        count_price = []
        for i, price in enumerate(price_range):
            if i == 0:
                continue
            val = basket_price[(basket_price['Basket Price'] < price) & (basket_price['Basket Price'] > price_range[i - 1])]['Basket Price'].count()
            count_price.append(val)

        def is_noun(pos):
            '''Noun validation'''
            return pos[:2] == 'NN'

        def keywords_inventory(dataframe, column='Description'):
            '''Stemming'''
            stemmer = nltk.stem.SnowballStemmer("english")
            keywords_roots = dict()  # collect the words / root
            keywords_select = dict()  # association: root <-> keyword
            category_keys = []
            count_keywords = dict()
            for s in dataframe[column]:
                if pd.isnull(s):
                    continue
                lines = s.lower()
                tokenized = nltk.word_tokenize(lines)
                nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]

                for t in nouns:
                    t = t.lower()
                    racine = stemmer.stem(t)
                    if racine in keywords_roots:
                        keywords_roots[racine].add(t)
                        count_keywords[racine] += 1
                    else:
                        keywords_roots[racine] = {t}
                        count_keywords[racine] = 1

            for s in keywords_roots.keys():
                if len(keywords_roots[s]) > 1:
                    min_length = 1000
                    for k in keywords_roots[s]:
                        if len(k) < min_length:
                            clef = k
                            min_length = len(k)
                    category_keys.append(clef)
                    keywords_select[s] = clef
                else:
                    category_keys.append(list(keywords_roots[s])[0])
                    keywords_select[s] = list(keywords_roots[s])[0]

            return category_keys, keywords_roots, keywords_select, count_keywords

        df_produits = pd.DataFrame(df_initial['Description'].unique()).rename(columns={0: 'Description'})

        keywords, keywords_roots, keywords_select, count_keywords = keywords_inventory(df_produits)

        list_products = []
        for k, v in count_keywords.items():
            list_products.append([keywords_select[k], v])
        list_products.sort(key=lambda x: x[1], reverse=True)

        liste = sorted(list_products, key=lambda x: x[1], reverse=True)

        list_products = []
        for k, v in count_keywords.items():
            word = keywords_select[k]
            if word in ['pink', 'blue', 'tag', 'green', 'orange']:
                continue
            if len(word) < 3 or v < 13:
                continue
            if ('+' in word) or ('/' in word):
                continue
            list_products.append([word, v])
        # ______________________________________________________
        list_products.sort(key=lambda x: x[1], reverse=True)

        liste_produits = df_cleaned['Description'].unique()
        X = pd.DataFrame()
        for key, occurence in list_products:
            X.loc[:, key] = list(map(lambda x: int(key.upper() in x), liste_produits))

        threshold = [0, 1, 2, 3, 5, 10]
        label_col = []
        for i in range(len(threshold)):
            if i == len(threshold) - 1:
                col = '.>{}'.format(threshold[i])
            else:
                col = '{}<.<{}'.format(threshold[i], threshold[i + 1])
            label_col.append(col)
            X.loc[:, col] = 0

        for i, prod in enumerate(liste_produits):
            prix = df_cleaned[df_cleaned['Description'] == prod]['UnitPrice'].mean()
            j = 0
            while prix > threshold[j]:
                j = j + 1
                if j == len(threshold):
                    break
            X.loc[i, label_col[j - 1]] = 1

        for i in range(len(threshold)):
            if i == len(threshold) - 1:
                col = '.>{}'.format(threshold[i])
            else:
                col = '{}<.<{}'.format(threshold[i], threshold[i + 1])

        matrix = X.to_numpy()  # as_matrix()
        for n_clusters in range(3, 10):
            kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=30)
            kmeans.fit(matrix)
            clusters = kmeans.predict(matrix)
            silhouette_avg = silhouette_score(matrix, clusters)

        n_clusters = 5
        silhouette_avg = -1
        while silhouette_avg < 0.145:
            kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=30)
            kmeans.fit(matrix)
            clusters = kmeans.predict(matrix)
            silhouette_avg = silhouette_score(matrix, clusters)

        pd.Series(clusters).value_counts()

        # define individual silouhette scores
        sample_silhouette_values = silhouette_samples(matrix, clusters)

        liste = pd.DataFrame(liste_produits)
        liste_words = [word for (word, occurence) in list_products]

        occurence = [dict() for _ in range(n_clusters)]

        for i in range(n_clusters):
            liste_cluster = liste.loc[clusters == i]
            for word in liste_words:
                if word in ['art', 'set', 'heart', 'pink', 'blue', 'tag']:
                    continue
                occurence[i][word] = sum(liste_cluster.loc[:, 0].str.contains(word.upper()))

        pca = PCA()
        pca.fit(matrix)
        pca_samples = pca.transform(matrix)

        pca = PCA(n_components=50)
        matrix_9D = pca.fit_transform(matrix)
        mat = pd.DataFrame(matrix_9D)
        mat['cluster'] = pd.Series(clusters)

        corresp = dict()
        for key, val in zip(liste_produits, clusters):
            corresp[key] = val
        # __________________________________________________________________
        df_cleaned['categ_product'] = df_cleaned.loc[:, 'Description'].map(corresp)

        for i in range(5):
            col = 'categ_{}'.format(i)
            df_temp = df_cleaned[df_cleaned['categ_product'] == i]
            price_temp = df_temp['UnitPrice'] * (df_temp['Quantity'] - df_temp['QuantityCanceled'])
            price_temp = price_temp.apply(lambda x: x if x > 0 else 0)
            df_cleaned.loc[:, col] = price_temp
            df_cleaned[col].fillna(0, inplace=True)
        # ____________________________________________________________________

        # sum of purchases / user & order
        temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['TotalPrice'].sum()
        basket_price = temp.rename(columns={'TotalPrice': 'Basket Price'})
        # ___________________________________________________________
        # percentage of order price / product category
        for i in range(5):
            col = 'categ_{}'.format(i)
            temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)[col].sum()
            basket_price.loc[:, col] = temp[col]


        # _____________________
        # date of order
        df_cleaned['InvoiceDate_int'] = df_cleaned['InvoiceDate'].astype('int64')
        temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate_int'].mean()
        df_cleaned.drop('InvoiceDate_int', axis=1, inplace=True)
        basket_price.loc[:, 'InvoiceDate'] = pd.to_datetime(temp['InvoiceDate_int'])
        # ______________________________________
        # selection of significant entries:
        basket_price = basket_price[basket_price['Basket Price'] > 0]

        set_entrainement = basket_price[basket_price['InvoiceDate'] < pd.to_datetime(datetime.date(2011, 10, 1))]
        set_test = basket_price[basket_price['InvoiceDate'] >= pd.to_datetime(datetime.date(2011, 10, 1))]
        basket_price = set_entrainement.copy(deep=True)

        # number of visits and stats on basket amount / users
        transactions_per_user = basket_price.groupby(by=['CustomerID'])['Basket Price'].agg(['count', 'min', 'max', 'mean', 'sum'])
        for i in range(5):
            col = 'categ_{}'.format(i)
            transactions_per_user.loc[:, col] = basket_price.groupby(by=['CustomerID'])[col].sum() / transactions_per_user['sum'] * 100

        transactions_per_user.reset_index(drop=False, inplace=True)
        basket_price.groupby(by=['CustomerID'])['categ_0'].sum()

        last_date = basket_price['InvoiceDate'].max().date()

        first_registration = pd.DataFrame(basket_price.groupby(by=['CustomerID'])['InvoiceDate'].min())
        last_purchase = pd.DataFrame(basket_price.groupby(by=['CustomerID'])['InvoiceDate'].max())

        test = first_registration.applymap(lambda x: (last_date - x.date()).days)
        test2 = last_purchase.applymap(lambda x: (last_date - x.date()).days)

        transactions_per_user.loc[:, 'LastPurchase'] = test2.reset_index(drop=False)['InvoiceDate']
        transactions_per_user.loc[:, 'FirstPurchase'] = test.reset_index(drop=False)['InvoiceDate']

        n1 = transactions_per_user[transactions_per_user['count'] == 1].shape[0]
        n2 = transactions_per_user.shape[0]

        list_cols = ['count', 'min', 'max', 'mean', 'categ_0', 'categ_1', 'categ_2', 'categ_3', 'categ_4']
        # ________________________________________________________
        selected_customers = transactions_per_user.copy(deep=True)
        matrix = selected_customers[list_cols].to_numpy()

        scaler = StandardScaler()
        scaler.fit(matrix)
        scaled_matrix = scaler.transform(matrix)

        pca = PCA()
        pca.fit(scaled_matrix)
        pca_samples = pca.transform(scaled_matrix)

        n_clusters = 11
        kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=100)
        kmeans.fit(scaled_matrix)
        clusters_clients = kmeans.predict(scaled_matrix)
        silhouette_avg = silhouette_score(scaled_matrix, clusters_clients)

        pca = PCA(n_components=6)
        matrix_3D = pca.fit_transform(scaled_matrix)
        mat = pd.DataFrame(matrix_3D)
        mat['cluster'] = pd.Series(clusters_clients)

        sample_silhouette_values = silhouette_samples(scaled_matrix, clusters_clients)
        # define individual silouhette scores
        sample_silhouette_values = silhouette_samples(scaled_matrix, clusters_clients)

        selected_customers.loc[:, 'cluster'] = clusters_clients

        merged_df = pd.DataFrame()
        for i in range(n_clusters):
            test = pd.DataFrame(selected_customers[selected_customers['cluster'] == i].mean(numeric_only=True))
            test = test.T.set_index('cluster', drop=True)
            test['size'] = selected_customers[selected_customers['cluster'] == i].shape[0]
            merged_df = pd.concat([merged_df, test])
        # _____________________________________________________

        merged_df = merged_df.sort_values('sum')

        liste_index = []
        for i in range(5):
            COLUMN = f'categ_{i}'
            liste_index.append(merged_df[merged_df[COLUMN] > 45].index.values[0])
        # ___________________________________
        liste_index_reordered = liste_index
        liste_index_reordered += [s for s in merged_df.index if s not in liste_index]
        # __________________________________________________________
        merged_df = merged_df.reindex(index=liste_index_reordered)
        merged_df = merged_df.reset_index(drop=False)

        if dataaugfactor != 0:
            selected_customers = pd.concat([selected_customers] * dataaugfactor, ignore_index=True)
        else:
            selected_customers = pd.concat([selected_customers], ignore_index=True)

        # Save to a csv file
        p, file = os.path.split(rawtraindata)
        split_tup = os.path.splitext(file)
        file_name = split_tup[0]
        file_extension = split_tup[1]
        if dataaugfactor > 0:
            SUFFIX = f'_aug_{dataaugfactor}'
        else:
            SUFFIX = '_aug'
        newdatafile = os.path.join(p, file_name + SUFFIX + file_extension)
        logger.info(f'Saving final filtered data to a csv file {newdatafile}...')
        selected_customers.to_csv(newdatafile, index=False)
    else:
        selected_customers = pd.read_csv(finaltraindata, encoding="ISO-8859-1",
                                         dtype={'CustomerID': str, 'InvoiceID': str})

        columns = ['mean', 'categ_0', 'categ_1', 'categ_2', 'categ_3', 'categ_4']
        X = selected_customers[columns]
        Y = selected_customers['cluster']

        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.8)
        
        if inference == '0':
            # Regular Training / hyperparamter tuning & model saving
            if algorithm == 'knn':
                # '''Algorithm = KNeighborsClassifier'''
                logger.info('Running KNeighborsClassifier ...')
                knn = neighbors.KNeighborsClassifier(n_jobs=-1)
    
                if tuning is True:
                    stime = time.time()
                    parameters={'n_neighbors' : np.arange(1, 25, 1)}
                    estimator, model  = hyperparameter_tuning(algorithm=knn, param_grid=parameters , kFold=5)
                    logger.info(f'====> KNeighborsClassifier Training Time with hyperparameter tuning {time.time()-stime} secs')
                    save_model(estimator, modelname=knn_model) # model is saved & loaded using joblib
                    logger.info("KNeighborsClassifier model 'knn_model.joblib' is saved in: /model ")

                else:
                    if batch_size is None:
                        tuned_params = {'n_neighbors': 1}
                        tuned_model = knn
                        tuned_model.set_params(**tuned_params)
                        stime = time.time()
                        tuned_model.fit(X_train, Y_train)

                        logger.info(f'====>KNeighborsClassifier Average Training Time with default hyperparameters {time.time()-stime} secs')

                if tuning is not True and batch_size is not None:

                    batchiterator = iter_minibatches(chunksize=batch_size)
                    # Tuned hyper parameter training for KNN
                    tuned_params = {'n_neighbors': 1}
                    tuned_model = neighbors.KNeighborsClassifier(n_jobs = -1)
                    tuned_model.set_params(**tuned_params)
                    total_time = 0
                    total_batches = 0
                    for X_batch, Y_batch in batchiterator:
                        stime = time.time()
                        tuned_model.fit(X_batch, Y_batch)
                        total_time += time.time() - stime
                        total_batches += 1

                    logger.info(f'====> KNeighborsClassifier Training Time with batch size {batch_size} is {total_time} secs')
                    logger.info(f'====>Average Training Time for {total_batches} batches is {total_time/total_batches} secs')
    
            elif algorithm == 'dtc':
                # '''Algorithm = DecisionTreeClassifier'''
                logger.info('Running DecisionTreeClassifier ...')
                dtc = tree.DecisionTreeClassifier()
    
                if tuning is True:
                    stime = time.time()
                    parameters={'criterion': ['entropy', 'gini'], 'max_features': ['sqrt', 'log2']}
                    estimator, model = hyperparameter_tuning(algorithm=dtc, param_grid=parameters , kFold=5)
                    logger.info(f'====> DecisionTreeClassifier Training Time with hyperparameter tuning {time.time()-stime} secs')
                    save_model(estimator, modelname=dtc_model) # model is saved & loaded using joblib
                    logger.info("DecisionTreeClassifier model 'dtc_model.joblib'is saved in: /model ")
                else:
                    stime = time.time()
                    dtc.fit(X_train, Y_train)
                    logger.info(f'====> DecisionTreeClassifier Training Time with default hyperparameters is {time.time()-stime} secs')
    
            elif algorithm == 'rfc':
                # '''Algorithm = RandomForestClassifier'''
                logger.info('Running RandomForestClassifier ...')
                rfc = ensemble.RandomForestClassifier(n_jobs=-1)
    
                if tuning is True:
                    #parameters = {'criterion': ['entropy', 'gini'], 'n_estimators': [20, 40, 60, 80, 100],
                    parameters = {'criterion': ['gini'],
                                  'n_estimators': [20, 40, 60, 80, 100],
                                  'max_features': ['sqrt', 'log2']}
                    stime = time.time()
                    estimator, model = hyperparameter_tuning(algorithm=rfc, param_grid=parameters, kFold=5)
                    logger.info(f'====> RandomForestClassifier Training Time with hyperparameter tuning {time.time()-stime} secs')
                    save_model(estimator, modelname=rfc_model)  # model is saved & loaded using joblib
                    logger.info("====> RandomForestClassifier model 'rfc_model.joblib'is saved in: /model ")

                else:
                    if batch_size is None:
                        # Tuned hyper parameter training for RFC
                        tuned_params = {'criterion': 'gini', 'max_features': 'log2', 'n_estimators': 100}
                        tuned_model_rf = rfc
                        tuned_model_rf.set_params(**tuned_params)
                        stime = time.time()
                        tuned_model_rf.fit(X_train, Y_train)
                        logger.info(f'====> RandomForestClassifier Training Time with default hyperparameters {time.time()-stime} secs')
                        #logger.info(f'====> RandomForestClassifier Training Time {time.time()-stime} secs')

                if tuning is not True and batch_size is not None:

                    batchiterator = iter_minibatches(chunksize=batch_size)
                    # Tuned hyper parameter training for RFC
                    tuned_params = {'criterion': 'gini', 'max_features': 'log2', 'n_estimators': 100}
                    tuned_model_rf = ensemble.RandomForestClassifier(n_jobs = -1)
                    tuned_model_rf.set_params(**tuned_params)
                    total_time = 0
                    total_batches = 0
                    for X_batch, Y_batch in batchiterator:
                        logger.info("Length X_batch", len(X_batch))
                        stime = time.time()
                        tuned_model_rf.fit(X_batch, Y_batch)
                        total_time += time.time() - stime
                        logger.info("total time is ", total_time)
                        total_batches += 1

                    logger.info(f'====>RandomForestClassifier Training Time with batch size {batch_size} is {total_time} secs')
                    logger.info(f'====>Average Training Time for {total_batches} batches is {total_time/total_batches} secs')

        elif inference == 'knn_model':
                X_test = X
                Y_test = Y
                loaded_model = load_model(knn_model)
                logger.info("kNN model loaded successfully")
                total_time = 0
                for i in range(100):
                    stime = time.time()
                    predictions = loaded_model.predict(X_test)
                    total_time += time.time() - stime
                logger.info(f'====> KNeighborsClassifier Model Average Inference Time is {total_time/100} secs')
                logger.info(f"====> Accuracy for kNN is: {100 * metrics.accuracy_score(Y_test, predictions)} % ")
                logger.info(f"====> F1 score for kNN is: { metrics.f1_score(Y_test,predictions,average = 'micro')}")
                
        elif inference == 'dtc_model':
                X_test = X
                Y_test = Y
                loaded_model = load_model(dtc_model)
                logger.info("Decision Tree Classifier model loaded successfully")
                stime = time.time()
                predictions = loaded_model.predict(X_test)
                logger.info(f'====> Decision Tree Classifier Model Inference Time is {time.time()-stime} secs')
                logger.info(f"====> Accuracy for DTC is: {100 * metrics.accuracy_score(Y_test, predictions)} % ")
                logger.info(f"====> F1 score for DTC is: { metrics.f1_score(Y_test,predictions,average = 'micro')}")
            
        elif inference == 'rfc_model':
                X_test = X
                Y_test = Y
                loaded_model= load_model(rfc_model)
                logger.info("Random Forest Classifier model loaded successfully")
                stime = time.time()
                predictions = loaded_model.predict(X_test)
                logger.info(f'====> Ramdom Forest Classifier Model Inference Time is {time.time()-stime} secs')
                logger.info(f"====> Accuracy for RFC is: {100 * metrics.accuracy_score(Y_test, predictions)} % ")
                logger.info(f"====> F1 score for RFC is: { metrics.f1_score(Y_test,predictions,average = 'micro')}")
        else:
            logger.info("====> Please check whether the correct model file name is passed or not!")
               
    logger.info(f'====> Program exeuction time {time.time()-prgstime} secs')
