from config import get_config_parser
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from climatenet.utils.metrics import get_iou_perClass
from os import path


def main(args):
    if args.train_path:
        train_df = pd.read_csv(path.join(args.train_path, 'trainSample.csv'))
        test_df = pd.read_csv(path.join(args.train_path, 'testSample.csv'))
    else:
        print('Missing training path')
        return
    
    if args.model_config is not None:
        print(f'Loading model config from {args.model_config}')
        with open(args.model_config) as f:
            model_config = json.load(f)
    else:
        raise ValueError('Please provide a model config json')

    features = ['PSL','TMQ','U850','V850']
    X_train = train_df[features].values
    y_train = train_df['LABELS'].values
    X_test = test_df[features].values
    y_test = test_df['LABELS'].values

    if args.standardize:
        train_avg = np.mean(X_train, axis=0)
        train_std = np.std(X_train, axis=0)
        X_train = (X_train - train_avg) / train_std
        X_test = (X_test - train_avg) / train_std

    model_cls = {'LogReg':LogisticRegression, 'XGBoost':XGBClassifier}[args.model]
    model = model_cls(**model_config)
    model.fit(X_train,y_train)
    preds = model.predict(X_test)
    cm = confusion_matrix(preds,y_test)

    logreg_acc_tt = model.score(X_test, y_test)
    logreg_acc_tr = model.score(X_train, y_train)
    ious = get_iou_perClass(cm)

    print("{} - Train accuracy: {:.4f}".format(args.model, logreg_acc_tr))
    print("{} - Test accuracy: {:.4f}".format(args.model, logreg_acc_tt))
    print("{} - IOUS: {}".format(args.model, ious))
    print("{} - mean IOU: {}".format(args.model, ious.mean()))
    

if __name__ == "__main__":
    parser = get_config_parser()
    args = parser.parse_args()
    main(args)
