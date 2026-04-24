4/25 我今天人工手动筛选了带式焙烧机在低负荷运转情况下生球进料为0左右甚至是开始减小或是增加总之不是正常的情况的数据，我进行了筛选。我采用了时间窗口法，由于FeO含量的采样率是六小时一次，而工况数据是每小时一次，我用六小时对应一小时的FeO ： 0:00 - 5:00  工况对应  6:00 FeO，以此类推。然后我进行了FeO时间序列的处理，它的行索引原来是日期 ,时间和FeO，我将其转为了正经的时间序列。芜湖然后我用箱型图对异常值进行了处理 IQR=Q3-Q1(第三分位数减去第一分位数) 再去除Q3+1.5IQR 和 Q1-1.5IQR以外的值替换NaN。总而言之，今天试了一下deepseek V4 的新模型，也是吃上细糠了。
明天待更
# Outlier detection 
import pandas as pd
import numpy as np
from collections import Counter
 
def detect_outliers(df,use_features): #异常值检索 
 
    outlier_indices = [] #IN Chinese, this means "异常值索引"
 

 
    # iterate over features(columns) in chinese, this means "遍历特征（列）"
    for col in use_features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
 
        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
 
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
 
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > 2 )
    return multiple_outliers   
这是异常值处理的代码
