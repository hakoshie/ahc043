import numpy as np
import re
import statsmodels.api as sm

def extract_data_from_file(filename):
    manhat, actual, ratio, turn, stations = [], [], [], [], []
    
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.search(r'manhat: (\d+) actual:([\d.]+) ratio:([\d.]+) turn:(\d+) stations:(\d+)', line)
            if match:
                manhat.append(int(match.group(1)))
                actual.append(float(match.group(2)))
                ratio.append(float(match.group(3)))
                turn.append(int(match.group(4)))
                stations.append(int(match.group(5)))  # 新しく追加

    return np.array(manhat), np.array(actual), np.array(ratio), np.array(turn), np.array(stations)

def calculate_correlation(filename):
    manhat, actual, ratio, turn, stations = extract_data_from_file(filename)
    data_matrix = np.array([manhat, actual, ratio, turn, stations])
    correlation_matrix = np.corrcoef(data_matrix)
    
    labels = ['manhat', 'actual', 'ratio', 'turn', 'stations']
    print("\nCorrelation Matrix:")
    print("\t" + "\t".join(labels))
    for i, row in enumerate(correlation_matrix):
        print(f"{labels[i]}\t" + "\t".join(f"{val:.3f}" for val in row))

def linear_regression_ratio_turn(filename):
    manhat, actual, ratio, turn, stations = extract_data_from_file(filename)
    
    # 説明変数（ターン数、駅数、マンハッタン距離、ターン数の2乗）
    X = np.column_stack((manhat,manhat**2))
    # X = manhat
    # X = sm.add_constant(X)  # 切片を追加

    # 目的変数（actual - manhat）
    y = actual  

    # 回帰モデルの作成と出力
    model = sm.OLS(y, X).fit()
    print("\nLinear Regression (actual - manhat -> turn, stations, manhat, turn^2):")
    print(model.summary())

if __name__ == "__main__":
    filename = "simulation.txt"  # 解析するファイルの名前
    calculate_correlation(filename)
    linear_regression_ratio_turn(filename)
