# -*- coding: utf-8 -*-

"""
Python 3
22 / 10 / 2025
@author: z_tjona


"I find that I don't understand things unless I try to program them."
-Donald E. Knuth

"Either mathematics is too big for the human mind or the human mind is more than a machine."
-Kurt Godël
"""
import pandas as pd

from tqdm import tqdm
import numpy as np


def calculate_BradleyTerry(
    score: dict[int, float],
    W: pd.DataFrame,
    EPOCHS: int = 4,
    diff_threshold: float = 1e-4,
    normalize: bool = False,
    verbose=False,
):
    """Calculate Bradley-Terry scores given initial scores and a win matrix W.
    # Arguments:
        score: (dict) Initial scores for each agent.
        W: A DataFrame where W.loc[i, j] is the number of wins of agent i over agent j.
        EPOCHS: Maximum number of iterations to perform.
        diff_threshold: Threshold for convergence based on the change in scores. If set to <=0, no convergence check is performed.
        normalize: If True, normalize scores after each epoch using geometric mean.
        verbose: If True, print progress information.
    """
    N_AGENTS = len(score)
    if verbose:
        print(f"Calculating Bradley-Terry for {N_AGENTS} agents.")

    for _ in range(EPOCHS):
        score_old = score.copy()
        for i in tqdm(score, desc="BT Epoch", disable=not verbose):
            _num, _den = 0, 0
            for j in score:
                if i == j:
                    continue
                wins_of_i_over_j = float(W.loc[i, j])  # type: ignore
                loses_of_i_to_j = float(W.loc[j, i])  # type: ignore

                _num += wins_of_i_over_j * score[j] / (score[i] + score[j])
                _den += loses_of_i_to_j / (score[i] + score[j])

            score[i] = _num / _den
        if normalize:
            # More stable version using logarithms
            log_scores = np.log(list(score.values()))
            log_mean = np.sum(log_scores) / N_AGENTS
            norm_factor = np.exp(log_mean)
            for i in score:
                score[i] /= norm_factor

        diff_norm2 = float("inf")  # Default value
        if diff_threshold > 0:
            diff_norm2 = (
                sum((score[i] - score_old[i]) ** 2 for i in score) ** 0.5 / N_AGENTS
            )
            if diff_norm2 <= diff_threshold:

                if verbose:
                    print(f"Diff norm2: {diff_norm2:.6f}")
                    print("Converged.")
                break

        if verbose:
            print(f"Diff norm2: {diff_norm2:.6f}")
            print([f"{x:.3f}" for x in score.values()])
    return score
