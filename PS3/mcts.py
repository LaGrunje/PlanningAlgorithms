#!/usr/bin/python

import numpy as np
from typing import Tuple
from utils import action_space, transition_function, pursuer_transition



#tree{ node: {parent: Tuple, reward: int, visits: int, children: List}}

def uct(node: Tuple, C=np.sqrt(0.1)) -> Tuple:
    children = tree[node]['children']
    children_weights = [(tree[child]['reward'] / tree[child]['visits']) + C * np.sqrt(np.log(tree[node]['visits']) / tree[child]['visits']) 
                        for child in children]
    return np.argmax(children_weights)
    


def selection(env: np.array, node_: Tuple, default_policy):
    reward = 0
    node = node_
    while True:
        x_e, _ = transition_function(env, node, action_space[default_policy[node]])
        
        if not x_e in tree:
            tree[node]['children'].append(x_e)
            tree[x_e] = {'parent': node, 'reward': 0, 'visits': 0, 'children': []}
            return x_e
        states = action_loop(env, node)
        for state in states:
            if not tuple(state) in tree:
                tree[node]['children'].append(tuple(state))
                tree[tuple(state)] = {'parent': node, 'reward': 0, 'visits': 0, 'children': []}
                return tuple(state)
        if len(tree[node]['children']) == 0:
            return
        else:
            child_reward = [tree[child]['reward'] for child in tree[node]['children']]
            ind = np.argmax(child_reward)
            node = tree[node]['children'][ind]

def simulate(env: np.array, x_e: Tuple, x_p: Tuple, goal: Tuple, k_budget, default_policy):
    reward = 0
    for i in range(k_budget):
        dist = np.linalg.norm(np.array(x_e) - np.array(goal))

        if x_e == goal:
            reward += 100
            return reward
        if x_e == x_p:
            return 0
        reward += 0.1/dist
        action = action_space[default_policy[x_e]]
        x_e, _ = transition_function(env, x_e, action)
        x_p = pursuer_transition(env, x_e, x_p)
    return reward

def backpropogation(node: Tuple, reward: int):
    while node != None:
        tree[node]['reward'] += reward
        tree[node]['visits'] += 1
        node = tree[node]['parent']

def mcts(env: np.array, x_e: Tuple, x_p: Tuple, goal: Tuple, k_budget, default_policy) -> Tuple:
    """
    Monte-Carlo tree search
    env is the grid enviroment
    x_e evader
    x_p pursuer
    goal is the goal state
    """
    node_ = selection(env, x_e, default_policy)
    if type(node_) == type((1,2)):
        reward = simulate(env, node_, x_p, goal, k_budget, default_policy)
        backpropogation(node_, reward)
    ind = uct(x_e)
    u = (tree[x_e]['children'][ind][0] - x_e[0], tree[x_e]['children'][ind][1] - x_e[1])
    return u
