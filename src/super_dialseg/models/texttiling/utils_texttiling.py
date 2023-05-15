import numpy as np


def depth_score_cal(scores):
    output_scores = []
    for i in range(len(scores)):
        lflag = scores[i]
        rflag = scores[i]
        if i == 0:
            hl = scores[i]
            for r in range(i + 1, len(scores)):
                if rflag <= scores[r]:
                    rflag = scores[r]
                else:
                    break
        elif i == len(scores):
            hr = scores[i]
            for l in range(i - 1, -1, -1):
                if lflag <= scores[l]:
                    lflag = scores[l]
                else:
                    break
        else:
            for r in range(i + 1, len(scores)):
                if rflag <= scores[r]:
                    rflag = scores[r]
                else:
                    break
            for l in range(i - 1, -1, -1):
                if lflag <= scores[l]:
                    lflag = scores[l]
                else:
                    break
        depth_score = 0.5 * (lflag + rflag - 2 * scores[i])
        output_scores.append(depth_score)

    return output_scores, np.mean(output_scores), np.std(output_scores)


def cutoff_threshold(cut_rate, mean, std):
    return mean + cut_rate * std

