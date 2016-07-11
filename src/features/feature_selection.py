from common import *
import numpy as np

# feature 名称及其在矩阵中的索引
g_feature_info = dict()

# 矩阵中每个feature 的重要性
g_features_importance = []

g_min_inportance = 0.01

def accumulateFeatureImportance(feature_importances_):
    global g_features_importance

    if (len(g_features_importance) == 0):
        g_features_importance.extend(feature_importances_)
    else:
        g_features_importance += feature_importances_

    return 0

def loggingFeatureImportance():
    logging.info("After split window, g_features_importance is : ")
    for feature_name, idx in g_feature_info.items():
        logging.info("%s : %.4f" % (feature_name, g_features_importance[idx]))

    return     

# 判断子特征矩阵中的特征有哪些是有效的
def featuresForForecasting(features_names, final_feature_importances):
    useful_features = []
    # logging.info("g_features_importance %s" % g_features_importance)
    # logging.info("g_feature_info %s" % g_feature_info)
    # logging.info("features_names %s" % features_names)

    for i, name in enumerate(features_names):
        if (final_feature_importances[g_feature_info[name]] >= g_min_inportance):
            useful_features.append(i)
            logging.info("feature (%s, %.4f) is usefull" % (name, final_feature_importances[g_feature_info[name]]))

    return useful_features


def getUsefulFeatures(during_training, cur_total_feature_cnt, feature_mat, features_names, useful_features):
    # 若是在训练过程中， 则保留子特征矩阵的所有列, 返回所有的特征名， 并记录下它们在特征矩阵中的索引
    if (during_training):
        for i, name in enumerate(features_names):
            g_feature_info[name] = cur_total_feature_cnt + i
        return feature_mat, len(features_names)
    else:
        # 不是在训练过程中（在预测过程中）， useful_features 指明了子特征矩阵中的哪些特征是有效的，只返回那些有效的子特征        
        return feature_mat[:, useful_features], len(useful_features)
