import pandas as pd
import numpy as np
import 最终代码
import time


df = pd.DataFrame.from_dict(最终代码.merged_dict, orient='index')
df[1] = 2 * (1 - df[1])

def normalize_data(df):
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df


# 标准化数据
normalized_df = normalize_data(df)


def entropy_weight(df):
    # 计算每个指标的熵
    p = df / df.sum()
    p_log = np.log(p)
    entropy = -p * p_log
    entropy_sum = entropy.sum()

    # 计算每个指标的权重
    weight = (1 - entropy) / (len(df) - entropy_sum)

    return weight


# 假设dataframe为df，包含多个评价指标
weights = entropy_weight(df)

# 找出每列的最大值和最小值
PIS = normalized_df.max()
NIS = normalized_df.min()


def calculate_distance(df, PIS, NIS):
    D_plus = np.sqrt(((df - PIS) ** 2).sum(axis=1))
    D_minus = np.sqrt(((df - NIS) ** 2).sum(axis=1))
    return D_plus, D_minus


D_plus, D_minus = calculate_distance(normalized_df, PIS, NIS)


def calculate_topsis_value(D_plus, D_minus):
    TOPSIS_value = D_minus / (D_plus + D_minus)
    return TOPSIS_value


TOPSIS_values = calculate_topsis_value(D_plus, D_minus)
sorted_TOPSIS_values = TOPSIS_values.sort_values(ascending=False)
# print(sorted_TOPSIS_values)

# 取前5%
# seed_nodes = list(sorted_TOPSIS_values.index)[:math.ceil(len(sorted_TOPSIS_values)*0.05)]
seed_nodes = list(sorted_TOPSIS_values.index)
# print(seed_nodes[:1634])
