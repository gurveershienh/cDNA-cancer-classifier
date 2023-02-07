import os
import pickle
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import LeaveOneOut

##custom functions for training and validation
def hold1outCV(estimator, X, y):
    '''
    Custom function for hold-one-out cross validation.
        estimator: sklearn estimator object
        X: np.array
        y: np.array
    
    '''
    leave_1_out = LeaveOneOut()

    results = {
        'TP': 0,
        'TN': 0,
        'FP': 0,
        'FN': 0
        }

    for train_ind, test_ind in leave_1_out.split(X):

        X_train, X_test = X[train_ind], X[test_ind]
        y_train, y_test = y[train_ind], y[test_ind]

        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        
        if y_pred == 1 and y_test == 1:
            results['TP'] += 1
        elif y_pred == 0 and y_test == 1:
            results['FN'] += 1
        elif y_pred == 1 and y_test == 0:
            results['FP'] += 1
        else:
            results['TN'] += 1
        
    return results



def abs_variance_feature_sel(pos_data,neg_data, ref_ids, threshold):
    '''
    Determines absolute variance for each feature between positive and negative samples
    and selects threshold number of features with the top variances
    
        pos_data: pd.DataFrame
        neg_data: pd.DataFrame
        ref_ids: pd.DataFrame.columns
        threshold: Int < len(ref_ids)
    '''
    ref_ids = list(ref_ids)
    
    variances = np.array([])
    
    for ref in ref_ids:
        pos_mean = pos_data[ref].mean()
        pos_stdev = pos_data[ref].std()
        neg_mean = neg_data[ref].mean()
        neg_stdev = neg_data[ref].std()
        
        var = abs(
            (pos_mean-neg_mean)/
            (pos_stdev-neg_stdev)
            )
        variances = np.append(variances, [var])
        
    top_var_ind = list(np.argpartition(variances, -threshold)[-threshold:])
    selected_feats = [ref_ids[i] for i in top_var_ind]
    selected_feats.sort()
    
    return selected_feats

# function for processing/loading raw cDNA data 
def load_data(folder, make_pkl=False):
    '''
    Loads cDNA gene expression data obtained from ArrayExpress database (Saintigny et al., 2011)
    and processes it into a pd.DataFrame
        folder: Str
        make_pkl" Boolean
    '''
    #set variables for dynamic file name strings
    neg_path = folder + '/1-GSM652{}.txt'
    neg_start = 764
    neg_end = 845

    pos_path = folder + '/GSM652{}_sample_table.txt'
    pos_start = 763
    pos_end = 847

    ##initialize first pos/neg cDNA sample into pd.Dataframe
    neg_data = pd.read_csv(neg_path.format(neg_start), sep='\t')
    pos_data = pd.read_csv(pos_path.format(pos_start), sep='\t')
    
    columns = [str(ref) for ref in neg_data.ID_REF]

    neg_df = pd.DataFrame([list(neg_data.VALUE)], columns=columns)
    pos_df = pd.DataFrame([list(pos_data.VALUE)], columns=columns)

    ##load rest of the negative samples
    idx = 1
    idx += neg_start
    while idx < neg_end:
        file = neg_path.format(idx)
        if os.path.isfile(file):
            data = pd.read_csv(file, sep='\t')
            temp_df = pd.DataFrame([list(data.VALUE)], columns=columns)
            neg_df = pd.concat((neg_df, temp_df), axis=0)
        idx += 1

    ##load rest of the positive samples
    idx = 1
    idx += pos_start
    while idx < pos_end:
        file = pos_path.format(idx)
        if os.path.isfile(file):
            data = pd.read_csv(file, sep='\t')
            temp_df = pd.DataFrame([list(data.VALUE)], columns=columns)
            pos_df = pd.concat((pos_df, temp_df), axis=0)
        idx += 1
    
    pos_df['label'] = [1] * len(pos_df)
    neg_df['label'] = [0] * len(neg_df)
    
    full_df = pd.concat((pos_df,neg_df), axis=0)
    full_df = full_df.loc[:,~full_df.columns.duplicated()].copy()
    
    if make_pkl is True:
        pickle.dump(full_df, open('data_obj.pkl', 'wb'))

    return (neg_df,pos_df,full_df)


def main():
    
    # call function for loading data
    neg_df, pos_df, df = load_data(folder='gene_exps')
    ref_ids = list(df.columns)
    
    # feature threshold
    th_lst = [50, 100, 500]
    
    # SVM diagonal factors
    c_lst = [1, 2, 5, 10, 100]

    # training model and evaluating metrics
    for th in th_lst:
        for c in c_lst:
            
            # call feature selection function
            feature_select = abs_variance_feature_sel(
                neg_data=neg_df,
                pos_data=pos_df,
                ref_ids=ref_ids,
                threshold=th
            )


            feature_selected_df = df[feature_select]
            X = feature_selected_df.drop(['label'], axis=1).to_numpy()
            y = feature_selected_df['label'].to_numpy()


            X, y = shuffle(X, y, random_state=999)

            # train and evaluate metrics
            estimator = SVC(C=c)
            model_metrics = hold1outCV(estimator,X,y)
            print(f'Model Metrics for DF={c} & num_features={th}\n{model_metrics}\n')
            
            return


if __name__ == '__main__':
    main()
    
    
    
    '''
-----------------------------------------------------------------------------------------------------------------------------------------------

    Model Metrics for DF=100 & num_features=500
    {'TP': 31, 'TN': 34, 'FP': 7, 'FN': 4}

    Use these values to build trained model for application. Code is commented out below.
    
    Automated parameter selection can be applied as well.
-----------------------------------------------------------------------------------------------------------------------------------------------
    '''
    
    # svm_model = SVC(C=100)

    # features = abs_variance_feature_sel(
    #     neg_data=neg_df,
    #     pos_data=pos_df,
    #     ref_ids=ref_ids,
    #     threshold=500
    # ) + ['label']

    # fs_df = full_df[features]
    # X, y = np.array(fs_df.drop('label', axis=1)), np.array(fs_df['label'])

    # svm_model.fit(X,y)

    # pickle.dump(svm_model, open('svm_model.pkl', 'wb'))