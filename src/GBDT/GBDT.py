from common import *
import numpy as np
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier

from feature_selection import *

import os



# 滑动窗口， window_end_date 为 Y， 从 [window_start_date, window_end_date -1] 范围内得到特征矩阵， 通过
# 训练得到特征的 importance
def GBDT_slideWindows(window_start_date, window_end_date, nag_per_pos, n_estimators, max_depth, 
                      learning_rate, min_samples_split, start_from, user_cnt):
    print("%s slide windows date (%s, %s), user (%d, %d)" % 
          (getCurrentTime(), window_start_date, window_end_date, start_from, user_cnt))

    print("        %s taking samples for slide window (%s, %d)   \r" % (getCurrentTime(), window_end_date, nag_per_pos))
    samples, Ymat = takeSamples(window_start_date, window_end_date, nag_per_pos, start_from, user_cnt)
    if (len(samples) == 0):
        print("%s No buy records from %s to %s, returning...   \r" % (getCurrentTime(), window_start_date, window_end_date))
        return None

    Xmat = createFeatureMatrix(window_start_date, window_end_date, nag_per_pos, samples)

    m, n = np.shape(Xmat)

    Xmat, Ymat, samples = shuffle(Xmat, Ymat, samples, random_state=13)

    params = {'n_estimators': n_estimators, 
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'learning_rate': learning_rate, 
              #'loss': 'ls'
              'loss': 'deviance'
              }

    #clf = GradientBoostingRegressor(**params)
    clf = GradientBoostingClassifier(**params)

    clf.fit(Xmat, Ymat)

    feature_importance = clf.feature_importances_

    logging.info("slide window [%s, %s], features (%d, %d) " % (window_start_date, window_end_date, m, n))

    return Xmat, Ymat, clf