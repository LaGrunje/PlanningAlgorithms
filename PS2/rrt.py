from angle_util import angle_difference, angle_linspace
from typing import List, Callable
import numpy as np

from environment import State, ManipulatorEnv

import matplotlib.pyplot as plt
import matplotlib.animation as animation


class RRTPlanner:
    
    def __init__(self,
                 env: ManipulatorEnv,
                 distance_fn: Callable,
                 max_angle_step: float = 10.0):
        """
        :param env: manipulator environment
        :param distance_fn: function distance_fn(state1, state2) -> float
        :param max_angle_step: max allowed step for each joint in degrees
        """
        self._env = env
        self._distance_fn = distance_fn
        self._max_angle_step = max_angle_step

    def random_sample(self, goal_state:  State) -> State:
        k = np.random.rand() * 100
        if k < 5:
            return goal_state
        else:
            return State(np.random.rand(4) * 360 - np.ones(4) * 180)

    def nearest(self, state: State, tree: List[object]) -> State:
        distance_list = [self._distance_fn(state, node[1]) for node in tree]
        return tree[min(range(len(distance_list)), key=distance_list.__getitem__)]

    def extend(self, state: State, random_state: State, step: float) -> State:
        rotation = angle_linspace(state.angles, random_state.angles, step)
        rotation = rotation[1]
        max_value = np.abs(rotation).max()
        counter = 1
        if 5 / max_value > 5:
            counter = 5 / max_value
        return State(rotation * counter)

        

    def generate_random_node(self, tree: List[object], path: List[object], vertices, goal_state: State):
        point_with_collisions = True
        node_name = f"q{len(tree)}"

        random_state = self.random_sample(goal_state)
        parent = self.nearest(random_state, tree)
        node = self.extend(parent[1], random_state, 10)

        if self._env.check_collision(node):
            return

        if node in vertices:
            return
        vertices.add(node)
        tree.append([node_name, node, parent[0]])
        path.append([parent[1], node])

    def check_path(self, tree: List[object], path: List[object], goal_state: State) -> bool:
        if self._distance_fn(goal_state, tree[-1][1]) < 10:
            path.append([tree[-1][1], goal_state])
            tree.append([f'q{len(tree)}', goal_state, tree[-1][0]])
            return True
        return False

    def plot(self, path: List[object], save_path: str = 'rod_solve.mp4'):
        imgs = []

        for s in path:
            self._env.state = s
            f = self._env.environment_image()
            s, (width, height) = f.canvas.print_to_buffer()
            X = np.frombuffer(s, np.uint8).reshape((height, width, 4))
            plt.clf()
            imgs.append(X)


        fig = plt.figure()
        ims = []
        for img in imgs:
            plot = plt.imshow(img)
            ims.append([plot])

        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)

        ani.save(save_path)

        plt.show()
        

    def plan(self,
             start_state: State,
             goal_state: State) -> List[State]:
        tree = []
        tree.append(['q0', start_state, 'None'])
        path = []
        vertices = set()
        vertices.add(start_state)
        while True:
            self.generate_random_node(tree, path, vertices, goal_state)
            if self.check_path(tree, path, goal_state):
                break
        print(len(tree))
        path = create_path(path, start_state, goal_state)
        print(len(path))
        self.plot(path)


def create_path(path: List[object], start_state: State, goal_state: State):
    ind = len(path) - 1
    res = []
    current_state = goal_state
    while ind >= 0 and path[ind][0] != start_state:
        if path[ind][1] == current_state:
            res.append(current_state)
            current_state = path[ind][0]
        ind -= 1
    res.append(start_state)
    res.reverse()
    return res


def L1(q1: State,
       q2: State) -> float:
    """
    :param q1: first configuration of the robot
    :param q2: second configuration of the robot
    :return: L1 distance between two configurations
    """
    return np.sum(np.absolute(angle_difference(q1.angles, q2.angles)))

def L1_weighted(q1: State,
       q2: State) -> float:
    """
    :param q1: first configuration of the robot
    :param q2: second configuration of the robot
    :return: weighted L1 distance
    """
    array = np.absolute(angle_difference(q1.angles, q2.angles))
    for i in range(4):
      array[i] = array[i] * (4 - i)
    return np.sum(array)
