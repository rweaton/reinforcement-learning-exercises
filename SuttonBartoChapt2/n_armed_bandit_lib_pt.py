import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from typing import Dict, List, Union, Callable, Tuple, Any
from importlib import reload


def stochastic_chooser(
    probs: Union[np.ndarray, torch.Tensor],
    n_picks:int, 
    bin_bounds: str = 'ceiling_inclusive',
    device: torch.device=torch.device("cpu")
) -> torch.Tensor:
    
    if isinstance(probs, np.ndarray):
        probs = torch.tensor(
            probs,
            dtype=torch.float32,
            device=device
        )
    indices = torch.arange(0, probs.shape[0], device=device)
    cum_probs_ceilings = torch.cumsum(probs, 0)
    cum_probs_floors = torch.cumsum(
        torch.hstack([
            torch.tensor([0.0], device=device),
            probs[:-1]
        ]), 0
    )
    seeds = torch.rand(n_picks, device=device)
    
    picks = torch.nan * torch.ones(n_picks, device=device)
    for i, seed in enumerate(seeds):
        if bin_bounds == 'floor_inclusive':
            ceiling_check = torch.logical_or(
                (cum_probs_ceilings > seed),
                (1.0*np.ones_like(cum_probs_ceilings) <= seed)
            )
            floor_check = (cum_probs_floors <= seed)
            select_mask = np.logical_and(ceiling_check, floor_check)
            pick = indices[select_mask]

        if bin_bounds == 'ceiling_inclusive':
            floor_check = torch.logical_or(
                (cum_probs_floors < seed),
                (0.0 * torch.ones_like(cum_probs_floors) >= seed)
            )
            ceiling_check = (cum_probs_ceilings >= seed)
            select_mask = torch.logical_and(ceiling_check, floor_check)
            pick = indices[select_mask]
            
        picks[i] = pick
        
    return torch.tensor(picks, dtype=torch.int32)

# TODO: Convert to pytorch
def stochastic_reward_gen(mean_reward_value: float, standard_dev: float) -> float:
    
    seed = np.random.rand(1)
    # Inverse of the cumulative normal distribution at specified mean and standard_deviation
    reward = stats.norm.ppf(seed, loc=mean_reward_value, scale=standard_dev)
    
    return reward


def action_value_estimator(reward: float, last_estimate: float, k_step: int) -> float:
    
    if k_step < 1:
        updated_estimate = last_estimate
    else:
        k_step = float(k_step)
        updated_estimate = last_estimate + (1. / (k_step + 1.)) * (reward - last_estimate)
    
    return updated_estimate


def alpha_k(k_step: int, exponent: float=1.) -> float:
    return (1. / (1 + k_step)) ** exponent


def action_value_estimator_alpha(
    reward: float, 
    last_estimate: float,
    k_step: int,
    alpha: Union[float, Callable],
    alpha_params: Tuple[Any]=(1.2,)
) -> float:
    
    if k_step < 1:
        updated_estimate = last_estimate
        
    if isinstance(alpha, float):
        updated_estimate = last_estimate + alpha * (reward - last_estimate)
        
    if isinstance(alpha, Callable):
        updated_estimate = last_estimate + alpha(k_step, *alpha_params) * (reward - last_estimate)
        
    return updated_estimate

# TODO: Convert to pytorch
def epsilon_greedy_actor(
    action_value_estimates: Union[np.ndarray, torch.Tensor],
    epsilon: float
) -> int:
    
    if isinstance(action_value_estimates, np.ndarray):
        action_value_estimates = torch.tensor(
            action_value_estimates, dtype=torch.float32
        )

    n_actions = action_value_estimates.shape[0]
    action_indices = torch.arange(0, n_actions)
    seed = torch.rand(1)
    if seed < epsilon:
        action_index = np.random.choice(action_indices, size=1, replace=True)  # need to convert to a pytorch equivalent !
    else:
        action_index = torch.argmax(action_value_estimates)
        
    return action_index

# TODO: Convert to pytorch
def softmax_actor(action_value_estimates: np.ndarray, tau: float) -> int:

    denom = np.sum(np.exp(action_value_estimates / tau))
    numer = np.exp(action_value_estimates / tau)
    
    probs = (1. / denom) * numer
    #print(f"value of probs: {probs}")
    action_index = stochastic_chooser(probs, n_picks=1, bin_bounds='ceiling_inclusive')
    
    return action_index

# TODO: Convert to pytorch
def draw_random_from_normal(n_actions: int, mean: float, std_dev: float) -> np.ndarray:
    
    seeds = np.random.rand(n_actions)
    drawn_values = np.array([stats.norm.ppf(seed, loc=mean, scale=std_dev) for seed in seeds])
    
    return drawn_values

# TODO: Convert to pytorch
def n_armed_bandit_solve(
    n_arms: int,
    n_runs: int,
    n_steps_per_run: int,
    actor: dict,
    action_value_estimation: Dict[str, Any],
    **kwargs
) -> list:
    
    mean_action_values_init = 0.
    stdev_action_values_init = 1.
    stdev_rewards = 1.
    runs_record = [{}] * n_runs
    new_action_value_estimates = np.random.rand(n_arms)
    
    action_value_estimator = action_value_estimation["function"]
    action_value_estim_params = action_value_estimation["params"]
    
    # Run loop
    for run_i in range(0, n_runs):
        action_record = np.nan*np.ones(n_steps_per_run)
        reward_record = np.nan*np.ones(n_steps_per_run)

        if "non_stationary" in  kwargs.keys():
            action_values_actual = 0.5 * np.ones(n_arms)

            best_actions = np.nan * np.ones(n_steps_per_run)
        else:
            action_values_actual = draw_random_from_normal(
                n_arms,
                mean_action_values_init,
                stdev_action_values_init,
            )
        
            best_actions = np.argmax(action_values_actual) * np.ones(n_steps_per_run)
            # print(f"Best action: {best_action}")
            # print(f"Action values actual: {action_values_actual}")

        #last_action_value_estimates = draw_random_from_normal(n_arms, mean_actuals, stdev_actuals)
        last_action_value_estimates = np.zeros_like(action_values_actual)
        action_select = np.random.choice(np.arange(0, n_arms))
        action_record[0] = action_select
        reward_record[0] = stochastic_reward_gen(
            action_values_actual[action_select],
            stdev_rewards,
        )
        
        # Play loop
        for step_i in range(0, n_steps_per_run - 1):

            step_k = np.sum(action_record == action_select)
            new_action_value_estimates[action_select] = action_value_estimator(
                reward_record[step_i],
                last_action_value_estimates[action_select],
                step_k,
                alpha=action_value_estim_params[0],
                alpha_params=action_value_estim_params[1:]
            )
            
            if actor["function"] == "epsilon-greedy":
                action_select = epsilon_greedy_actor(
                    new_action_value_estimates, *actor["params"]
                )
                
            if  actor["function"] == "softmax_actor":
                action_select = softmax_actor(
                    new_action_value_estimates, *actor["params"]
                )
            
            action_record[step_i + 1] = action_select

            if "non_stationary" in  kwargs.keys():
                # Displace actual action values by random walk iteration
                beta = kwargs["non_stationary"]
                # action_values_actual += beta * (np.random.rand(n_arms) - 0.5)
                # Round to nearest integer in {-1, 0, 1}
                action_values_actual += beta * np.round(2. * (np.random.rand(n_arms) - 0.5))
                # Select element in {-1, 1}
                # seeds = np.random.rand(n_arms)
                # action_values_actual[seeds >= 0.5] += beta 
                # action_values_actual[seeds < 0.5] -= beta
                # action_values_actual[action_values_actual < 0.] = 0.
                # action_values_actual[action_values_actual > 1.] = 1.
                best_actions[step_i + 1] = np.argmax(action_values_actual)
            
            reward_record[step_i + 1] = stochastic_reward_gen(
                action_values_actual[action_select], 
                stdev_rewards
            )

            last_action_value_estimates = new_action_value_estimates

        runs_record[run_i] = dict(
            best_actions=best_actions,
            action_record=action_record.astype(int),
            reward_record=reward_record
        )
        
    return runs_record


def runs_plotter(
    runs_record: List[dict],
    ax_optimal_choice: plt.axes,
    ax_reward: plt.axes,
    color: str="auto",
) -> None:
    
    n_runs = len(runs_record)
    
    steps_trace = np.arange(0, runs_record[0]["reward_record"].shape[-1])
    reward_trace = np.zeros_like(runs_record[0]["reward_record"])
    best_trace = np.zeros_like(runs_record[0]["action_record"])
    for i in range(0, n_runs):
        best_actions = runs_record[i]["best_actions"]
        reward_trace += runs_record[i]["reward_record"]
        best_trace += np.equal(runs_record[i]["action_record"], best_actions)

    best_trace = best_trace / n_runs
    reward_trace = reward_trace / n_runs
    
    ax_optimal_choice.plot(steps_trace, best_trace, c=color)
    ax_reward.plot(steps_trace, reward_trace, c=color)