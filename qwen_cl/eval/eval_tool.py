import re
import numpy as np



def edit_distance_accuracy(pred: str, label: str) -> float:
    """
    编辑距离准确度（保留：如果你后面还想用到）。现在主评测不再依赖它也没关系。
    """
    pred = pred or ""
    label = label or ""
    dist = levenshtein_distance(pred, label)
    acc = 1 - dist / len(label) if len(label) > 0 else 0.0
    return max(float(acc), 0.0)

def levenshtein_distance(s1: str, s2: str) -> int:
    n, m = len(s1), len(s2)
    dp = np.zeros((n + 1, m + 1), dtype=int)

    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,      # deletion
                dp[i, j - 1] + 1,      # insertion
                dp[i - 1, j - 1] + cost # substitution
            )
    return int(dp[n, m])
