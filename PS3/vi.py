#!/usr/bin/python

import numpy as np
from typing import Tuple
from utils import action_space, transition_function


def action_loop(env: np.array, state: Tuple) -> np.array:
    new_pos = []
    if env[state] >= 1.0 - 1e-4:
        return np.array([])
    for action in action_space:
        new_state, condition = transition_function(env, state, action)
        if condition:
            new_pos.append(new_state)
    return np.array(new_pos)

def vi_step(env: np.array, cost_to_go: np.array) -> np.array:
    cost_to_go_new = np.ones(env.shape) * 1e2
    
    return cost_to_go_new

def decrypt(step: Tuple) -> int:
    if step == (-1,0):
        return 0
    if step == (0,-1):
        return 1
    if step == (1, 0):
        return 2
    if step == (0, 1):
        return 3    


def vi(env: np.array, goal: Tuple) -> (np.array, np.array):
    """
    env is the grid enviroment
    goal is the goal state
    """
    policy, cost_to_go = np.zeros(env.shape, 'b'), np.ones(env.shape) * 1e2
    cost_to_go[goal[0], goal[1]] = 0
    while True:
        cost_to_go_prev = np.copy(cost_to_go)
        for i in range(env.shape[0]):
            for j in range(env.shape[1]):
                next_states = action_loop(env, (i, j))
                Q = []
                for next_state in next_states:
                    Q.append((next_state, 1 + cost_to_go[next_state[0], next_state[1]]))
                if(len(Q) == 0):
                    continue
                min_value = 100
                min_elem = Q[0][0]
                for k in Q:
                    if k[1] < min_value:
                        min_elem = k[0]
                        min_value = k[1]
                cost_to_go[i,j] = min(min_value, cost_to_go[i,j])
                policy[i][j] = decrypt((min_elem[0] - i, min_elem[1] -j))
        if np.sum(cost_to_go - cost_to_go_prev) == 0:
            break 
    return policy, cost_to_go
