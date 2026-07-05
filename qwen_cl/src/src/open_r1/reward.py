import numpy as np
import re
def levenshtein_distance(s1, s2):
    """
    计算编辑距离（Levenshtein距离）
    :param s1: 字符串1
    :param s2: 字符串2
    :return: 编辑距离
    """
    n, m = len(s1), len(s2)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    
    return dp[n][m]

def edit_distance_accuracy(pred, label):
    """
    计算编辑距离准确度
    :param pred: 模型输出的 LaTeX 公式字符串
    :param label: 真实标签的 LaTeX 公式字符串
    :return: 准确度
    """
    edit_distance = levenshtein_distance(pred, label)
    accuracy = 1 - edit_distance / len(label) if len(label) > 0 else 0.0
    return max(accuracy,0.0)

def contains_special_symbols(text):
    """
    如果字符串 text 中包含以下符号任意一个：+ = - \ { } ^ $
    则返回 True，否则返回 False
    """
    # 在正则字符类中列出待匹配的符号
    # 说明：
    #   - 反斜杠 \ 在 Python 字符串和正则表达式中都要进行转义
    #   - 大括号 { }、插入符 ^、美元符号 $、减号 - 等在正则表达式里
    #     具有特殊含义，需要转义或放在适当位置
    pattern = r'[+\=\-\\\{\}\^\$]'   
    
    return bool(re.search(pattern, text))
def find_words_num(text):
    words = re.findall(r'\b\w+\b', text)
    return words

if __name__ == "__main__":
    pred = '\\[\n\\alpha = -\\frac{1}{\\sqrt{12}}\n\\]'
    ground_truth = '\\alpha=-\\frac{1}{\\sqrt 12}'
    accuracy = edit_distance_accuracy(pred, ground_truth)
    print(accuracy)