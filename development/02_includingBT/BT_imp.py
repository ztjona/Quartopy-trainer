from math import prod
import pandas as pd


def calculate_BradleyTerry(
    score: dict[int, float],
    W: pd.DataFrame,
    EPOCHS: int = 4,
    diff_threshold: float = 1e-3,
    verbose=False,
):
    """Calculate Bradley-Terry scores given initial scores and a win matrix W.
    # Arguments:
        score: (dict) Initial scores for each agent.
        W: A DataFrame where W.iloc[i, j] is the number of wins of agent i over agent j.
        EPOCHS: Maximum number of iterations to perform.
        diff_threshold: Threshold for convergence based on the change in scores.
        verbose: If True, print progress information.
    """
    N_AGENTS = len(score)
    for _ in range(EPOCHS):
        # score_new = [0.0] * N_AGENTS
        score_old = score.copy()
        for i in score.__reversed__():
            _num, _den = 0, 0
            for j in score:
                if i == j:
                    continue
                wins_of_i_over_j = float(W.loc[i, j])  # type: ignore
                loses_of_i_to_j = float(W.loc[j, i])  # type: ignore

                _num += wins_of_i_over_j * score[j] / (score[i] + score[j])
                _den += loses_of_i_to_j / (score[i] + score[j])

            score[i] = _num / _den
            # score_new[i] = _num / _den

            # Normalize by geometric mean -- not used for numerical stability
            # _m_g = prod([x ** (1 / N_AGENTS) for x in score_new])
            # score_new = [x / _m_g for x in score_new]
        diff_norm2 = (
            sum((score[i] - score_old[i]) ** 2 for i in score) ** 0.5 / N_AGENTS
        )
        # if diff_norm2 < diff_threshold:
        #     break
        if verbose:
            print(f"Diff norm2: {diff_norm2:.6f}")
            print([f"{x:.3f}" for x in score])
    return score


import pickle
import time

f_name = "EXP_id03.pkl"
with open(f_name, "rb") as f:
    _results = pickle.load(f)

v = [
    -1,
] + list(range(1100))


# agent: -1 dummy player, everybody wins and loses against it
# agent 0 is the random agent at the start of training
scores = {-1: 1.0, 0: 1.0}  # initial scores
results = pd.DataFrame(0.0, index=v, columns=v)
results.loc[0, -1] = 1.0
results.loc[-1, 0] = 1.0

times_per_agent = []
for i_m1, _res in enumerate(_results):
    # add new agent to the pool
    i = i_m1 + 1  # index of the new agent
    if i % 10 == 0:
        print(f"Adding agent {i}| {times_per_agent[-1]:.2f}s")
        print(scores)
    scores[i] = 1.0
    # print(i, scores)
    # results.loc[i] = 0.0
    # results[i] = 0.0
    # add results of the new agent
    for j, _res_ij in _res.items():
        # print(_res_ij)
        results.loc[i, j] += _res_ij["wins"]
        results.loc[j, i] += _res_ij["losses"]

        # draws are half win and half loss
        results.loc[i, j] += _res_ij["draws"] / 2
        results.loc[j, i] += _res_ij["draws"] / 2
    results.loc[-1, i] = 1.0
    results.loc[i, -1] = 1.0

    start_time = time.time()
    # recalculate scores
    scores = calculate_BradleyTerry(scores, results, EPOCHS=2, verbose=False)
    times_per_agent.append(time.time() - start_time)

print(times_per_agent)
