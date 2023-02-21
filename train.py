from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from climatenet.utils.metrics import get_cm, get_iou_perClass
from os import path


def add_args(parser):
    """
    Add arguments to parser
    """
    parser.add_argument(
        "--train_path",
        default='data',
        type=str,
        help="Train Path",
    )
    return parser


def main(args):
    if args.train_path:
        train_df = pd.read_csv(path.join(args.train_path, 'trainSample.csv'))
        test_df = pd.read_csv(path.join(args.train_path, 'testSample.csv'))
    else:
        print('Missing training path')
        return

    features = ['PSL','TMQ','U850','V850']
    X_train = train_df[features].values
    y_train = train_df['LABELS'].values
    X_test = test_df[features].values
    y_test = test_df['LABELS'].values

    logreg = LogisticRegression(random_state=0, max_iter=500, verbose=1).fit(X_train, y_train)
    preds = logreg.predict(X_test)
    cm = confusion_matrix(preds,y_test)

    logreg_acc_tt = logreg.score(X_test, y_test)
    logreg_acc_tr = logreg.score(X_train, y_train)
    ious = get_iou_perClass(cm)

    print("LogReg - Train accuracy: {:.4f}".format(logreg_acc_tr))
    print("LogReg - Test accuracy: {:.4f}".format(logreg_acc_tt))
    print("LogReg - IOUS: {}".format(ious))
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)