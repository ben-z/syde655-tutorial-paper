#%%
import matplotlib.pyplot as plt
import numpy as np
import random
from gambling_problem_stochastic_dynamic_programming import GamblersRuin
from statistics import mean, stdev
from time import perf_counter

def bet(current_wealth,bet_amount,winning_prob):
    if winning_prob >= random.uniform(0, 1):
        return current_wealth + bet_amount
    else:
        return current_wealth - bet_amount


def play(betting_horizon, initial_wealth,winning_prob,betting_fn):
    wealth = [-1]*(betting_horizon+1)
    wealth[0] = initial_wealth
    bet_amounts = [-1]*betting_horizon
    for k in range(betting_horizon):
        bet_amounts[k] = betting_fn(k, wealth[k])
        wealth[k+1] = bet(wealth[k], bet_amounts[k], winning_prob)   
    
    return wealth, bet_amounts

def run_benchmark(betting_horizon,initial_wealth,target_wealth,winning_prob,num_games,methods=['dp','greedy']):
    pmf = [[(0, 1-winning_prob),(2, winning_prob)] for i in range(0,betting_horizon)]

    benchmark_result = PerfResult()
    benchmark_result.num_games = num_games
    # ensure that the random number generation is consistent between runs
    randseed = random.randint(0,10000)

    if 'dp' in methods or 'dp-theoretical' in methods:
        # initialize stochastic dynamic programming
        gr = GamblersRuin(betting_horizon, target_wealth, pmf)
        dp_theoretical_winning_prob = gr.f(initial_wealth)
        benchmark_result.dp_theoretical_winning_prob = dp_theoretical_winning_prob

    if 'dp' in methods:
        def dp_betting_fn(k, current_wealth):
            return gr.q(k, current_wealth)
        random.seed(randseed)
        dp_start = perf_counter()
        dp_wins = 0
        for _ in range(num_games):
            dp_wealth,_ = play(betting_horizon, initial_wealth, winning_prob, dp_betting_fn)
            dp_wins += dp_wealth[-1] >= target_wealth
        dp_stop = perf_counter()
        benchmark_result.dp_wins = dp_wins
        benchmark_result.dp_secs_elapsed = dp_stop - dp_start

    if 'greedy' in methods:
        def greedy_betting_fn(k, current_wealth):
            return max(0, min(current_wealth, target_wealth-current_wealth))

        random.seed(randseed)
        greedy_start = perf_counter()
        greedy_wins = 0
        for _ in range(num_games):
            greedy_wealth,_ = play(betting_horizon, initial_wealth, winning_prob, greedy_betting_fn)
            greedy_wins += greedy_wealth[-1] >= target_wealth
        greedy_stop = perf_counter()
        benchmark_result.greedy_wins = greedy_wins
        benchmark_result.greedy_secs_elapsed = greedy_stop - greedy_start


    return benchmark_result


class PerfResult:
    num_games = None
    dp_theoretical_winning_prob = None
    dp_wins = None
    dp_secs_elapsed = None
    greedy_wins = None
    greedy_secs_elapsed = None

#%%
# DP convergence
results = []
for num_games in [1, 10, 100, 1000, 10000]:
    proportion_of_wins = []
    for _ in range(100):
        benchmark_result = run_benchmark(betting_horizon=4,initial_wealth=2,target_wealth=6,winning_prob=0.4,num_games=num_games,methods=['dp'])
        proportion_of_wins.append(benchmark_result.dp_wins / benchmark_result.num_games)
    results.append([num_games,proportion_of_wins,benchmark_result.dp_theoretical_winning_prob])

plt.errorbar([r[0] for r in results], [mean(r[1]) for r in results], [stdev(r[1]) for r in results], linestyle='None', marker='^', capsize=3,label='experimental')
plt.plot([r[0] for r in results], [r[2] for r in results], label='theoretical')
plt.xlabel('Number of Games')
plt.ylabel('Proportion of Wins')
plt.title('Convergence of DP Solution to Theoretical')
plt.xscale('log')
plt.legend()
plt.savefig('dp-convergence.png')
plt.show()

#%%
# Variable horizon DP v.s. Greedy
results = []
for betting_horizon in range(2,20):
    results.append([betting_horizon,run_benchmark(betting_horizon=betting_horizon,initial_wealth=2,target_wealth=6,winning_prob=0.4,num_games=100000,methods=['greedy','dp-theoretical'])])

# plt.plot([r[0] for r in results], [r[1].dp_wins /
#                                    r[1].num_games for r in results], label='DP')
plt.plot([r[0] for r in results], [r[1].dp_theoretical_winning_prob for r in results], label='DP Theoretical')
plt.plot([r[0] for r in results], [r[1].greedy_wins/r[1].num_games for r in results], label='Greedy')
plt.xlabel('Betting Horizon')
plt.ylabel('Proportion of Wins')
plt.title('DP v.s. Greedy (Variable Horizon)')
plt.legend()
plt.savefig('dp-vs-greedy-variable-horizon.png')
plt.show()

#%%
# Variable target wealth DP v.s. Greedy
results = []
for target_wealth in range(11):
    results.append([target_wealth,run_benchmark(betting_horizon=4,initial_wealth=2,target_wealth=target_wealth,winning_prob=0.4,num_games=100000,methods=['greedy','dp-theoretical'])])

# plt.plot([r[0] for r in results], [r[1].dp_wins /
#                                    r[1].num_games for r in results], label='DP')
plt.plot([r[0] for r in results], [r[1].dp_theoretical_winning_prob for r in results], label='DP Theoretical')
plt.plot([r[0] for r in results], [r[1].greedy_wins/r[1].num_games for r in results], label='Greedy')
plt.xlabel('Target Wealth')
plt.ylabel('Proportion of Wins')
plt.title('DP v.s. Greedy (Variable Target Wealth)')
plt.legend()
plt.savefig('dp-vs-greedy-variable-target-wealth.png')
plt.show()

#%%
# Variable Per-Play Winning Probability DP v.s. Greedy
results = []
for winning_prob in np.arange(0,1.1,0.1):
    results.append([winning_prob, run_benchmark(betting_horizon=4, initial_wealth=2, target_wealth=6,
                                                winning_prob=winning_prob, num_games=100000, methods=['greedy', 'dp-theoretical'])])

# plt.plot([r[0] for r in results], [r[1].dp_wins /
#                                    r[1].num_games for r in results], label='DP')
plt.plot([r[0] for r in results], [r[1].dp_theoretical_winning_prob for r in results], label='DP Theoretical')
plt.plot([r[0] for r in results], [r[1].greedy_wins/r[1].num_games for r in results], label='Greedy')
plt.xlabel('Per-Play Winning Probability')
plt.ylabel('Proportion of Wins')
plt.title('DP v.s. Greedy (Variable Per-Play Winning Probability)')
plt.legend()
plt.savefig('dp-vs-greedy-variable-per-play-winning-prob.png')
plt.show()

#%%
# Variable Per-Play Winning Probability with betting horizon 10 DP v.s. Greedy
results = []
for winning_prob in np.arange(0,1.1,0.1):
    results.append([winning_prob, run_benchmark(betting_horizon=10, initial_wealth=2, target_wealth=6,
                                                winning_prob=winning_prob, num_games=100000, methods=['greedy', 'dp-theoretical'])])

# plt.plot([r[0] for r in results], [r[1].dp_wins /
#                                    r[1].num_games for r in results], label='DP')
plt.plot([r[0] for r in results], [r[1].dp_theoretical_winning_prob for r in results], label='DP Theoretical')
plt.plot([r[0] for r in results], [r[1].greedy_wins/r[1].num_games for r in results], label='Greedy')
plt.xlabel('Per-Play Winning Probability')
plt.ylabel('Proportion of Wins')
plt.title('DP v.s. Greedy (Variable Per-Play Winning Probability)')
plt.legend()
plt.savefig('dp-vs-greedy-variable-per-play-winning-prob-and-betting-horizon-10.png')
plt.show()

#%%
# Variable horizon + per-play winning probability 0.8 DP v.s. Greedy
results = []
for betting_horizon in range(2,20):
    results.append([betting_horizon,run_benchmark(betting_horizon=betting_horizon,initial_wealth=2,target_wealth=6,winning_prob=0.8,num_games=100000,methods=['greedy','dp-theoretical'])])

# plt.plot([r[0] for r in results], [r[1].dp_wins /
#                                    r[1].num_games for r in results], label='DP')
plt.plot([r[0] for r in results], [r[1].dp_theoretical_winning_prob for r in results], label='DP Theoretical')
plt.plot([r[0] for r in results], [r[1].greedy_wins/r[1].num_games for r in results], label='Greedy')
plt.xlabel('Betting Horizon')
plt.ylabel('Proportion of Wins')
plt.title('DP v.s. Greedy (Variable Horizon)')
plt.legend()
plt.savefig('dp-vs-greedy-variable-horizon-and-per-play-winning-prob-0_8.png')
plt.show()

# %%
# Computational Efficiency DP v.s. Greedy
results = []
for num_games in [1, 10, 100, 1000, 10000, 100000, 1000000]:
    results.append([num_games,run_benchmark(betting_horizon=4,initial_wealth=2,target_wealth=6,winning_prob=0.4,num_games=num_games)])

plt.plot([r[0] for r in results], [r[1].dp_secs_elapsed for r in results], marker='.', label='DP')
plt.plot([r[0] for r in results], [r[1].greedy_secs_elapsed for r in results], marker='.', label='Greedy')
plt.xlabel('Number of Games (Log Scale)')
plt.ylabel('Computation Time (s)')
plt.title('Computation Time v.s. Number of Games')
plt.legend()
plt.xscale('log')
plt.savefig('dp-vs-greedy-computation-time-log.png')
plt.show()

plt.plot([r[0] for r in results], [r[1].dp_secs_elapsed for r in results], marker='.', label='DP')
plt.plot([r[0] for r in results], [r[1].greedy_secs_elapsed for r in results], marker='.', label='Greedy')
plt.xlabel('Number of Games')
plt.ylabel('Computation Time (s)')
plt.title('Computation Time v.s. Number of Games')
plt.legend()
plt.savefig('dp-vs-greedy-computation-time.png')
plt.show()

# %%
# Computational Efficiency (Variable Horizon) DP v.s. Greedy
results = []
for betting_horizon in range(2,10):
    results.append([betting_horizon,run_benchmark(betting_horizon=betting_horizon,initial_wealth=2,target_wealth=6,winning_prob=0.4,num_games=100000)])

plt.plot([r[0] for r in results], [r[1].dp_secs_elapsed for r in results], marker='.', label='DP')
plt.plot([r[0] for r in results], [r[1].greedy_secs_elapsed for r in results], marker='.', label='Greedy')
plt.xlabel('Betting Horizon')
plt.ylabel('Computation Time (s)')
plt.title('Computation Time v.s. Betting Horizon')
plt.legend()
plt.savefig('dp-vs-greedy-computation-time-vs-betting-horizon.png')
plt.show()
# %%
