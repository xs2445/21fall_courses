import numpy as np
from RLalgs.utils import epsilon_greedy
import random

def QLearning(env, num_episodes, gamma, lr, e):
    """
    Implement the Q-learning algorithm following the epsilon-greedy exploration.

    Inputs:
    env: OpenAI Gym environment 
            env.P: dictionary
                    P[state][action] are tuples of tuples tuples with (probability, nextstate, reward, terminal)
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    num_episodes: int
            Number of episodes of training
    gamma: float
            Discount factor.
    lr: float
            Learning rate.
    e: float
            Epsilon value used in the epsilon-greedy method.

    Outputs:
    Q: numpy.ndarray
    """

    Q = np.zeros((env.nS, env.nA))
    
    #TIPS: Call function epsilon_greedy without setting the seed
    #      Choose the first state of each episode randomly for exploration.
    ############################
    # YOUR CODE STARTS HERE

    for i in range(num_episodes):
        env.isd = [1 / env.nS] * env.nS
        S = env.reset()
        if S not in ['5','7','11','12','15']:
            terminal_flag = False
            while not terminal_flag:
                # greedy strategy to choose the action
                action = epsilon_greedy(Q[S], e, seed = None)
                # take that action
                nextstate, r, terminal_flag, _ = env.step(action)
                # update q value using bellman optimility quation
                Q[S][action] += lr * (r + gamma * np.max(Q[nextstate]) - Q[S][action])
                # move to next state
                S = nextstate

    # YOUR CODE ENDS HERE
    ############################

    return Q