import subprocess
import numpy as np
import scipy.stats as stats

# プログラムを実行する関数
def run_pahcer():
    result = subprocess.run(["pahcer", "run"], capture_output=True, text=True)
    # print("Pahcer output:", result.stdout)  # 出力を確認
    return result.stdout


# 統計データを収集する関数
def collect_scores():
    scores = []
    for _ in range(3):  # 10回実行
        print(f"Running pahcer... iteration {_+1}")
        output = run_pahcer()
        # 出力からスコアを抽出する処理（出力形式に応じて変更）
        # 例として、"Average Score: xxx"と"Average Relative Score: xxx"から数値を取得する
        # Extract the average score
        average_score_str = output.split("Average Score          : ")[1].split()[0]
        
        # Remove commas and convert to float
        average_score = float(average_score_str.replace(',', ''))
        average_relative_score = float(output.split("Average Relative Score : ")[1].split()[0])
        scores.append((average_score, average_relative_score))
    return np.array(scores)

# メイン処理
def main():
    # tmp.cppをmain_tmp.cppにコピーして変更前のプログラムを準備
    subprocess.run(["cp", "main.cpp", "main_tmp.cpp"])
    # 変更前のスコアを収集
    scores_before = collect_scores()
    print("done")
    # tmp.cppをmain.cppにコピーして変更後のプログラムを準備
    subprocess.run(["cp", "tmp.cpp", "main.cpp"])
    print("done")
    # 変更後のスコアを収集
    scores_after = collect_scores()
    subprocess.run(["cp", "main_tmp.cpp", "main.cpp"])
    subprocess.run(["rm", "main_tmp.cpp"])
    print("done")
    # スコアを分離
    average_scores_before = scores_before[:, 0]
    average_relative_scores_before = scores_before[:, 1]
    average_scores_after = scores_after[:, 0]
    average_relative_scores_after = scores_after[:, 1]

    #print average average scores
    print("Average Score Before: {:.4f}".format(np.mean(average_scores_before)))
    print("Average Score After: {:.4f}".format(np.mean(average_scores_after)))
    print("Average Relative Score Before: {:.4f}".format(np.mean(average_relative_scores_before)))
    print("Average Relative Score After: {:.4f}".format(np.mean(average_relative_scores_after)))

    # t検定を実施
    t_stat, p_value_score = stats.ttest_ind(average_scores_before, average_scores_after)
    t_stat_rel, p_value_rel_score = stats.ttest_ind(average_relative_scores_before, average_relative_scores_after)

    # 結果を出力
    print("Average Score - t-statistic: {:.4f}, p-value: {:.4f}".format(t_stat, p_value_score))
    print("Average Relative Score - t-statistic: {:.4f}, p-value: {:.4f}".format(t_stat_rel, p_value_rel_score))

    # 有意差の判定
    alpha = 0.05
    if p_value_score < alpha:
        print("Average Scoreに有意な差があります。")
    else:
        print("Average Scoreに有意な差はありません。")

    if p_value_rel_score < alpha:
        print("Average Relative Scoreに有意な差があります。")
    else:
        print("Average Relative Scoreに有意な差はありません。")

if __name__ == "__main__":
    main()
