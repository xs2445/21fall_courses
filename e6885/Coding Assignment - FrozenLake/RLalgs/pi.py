import numpy as np
from RLalgs.utils import action_evaluation

def policy_iteration(env, gamma, max_iteration, theta):
    """
    Implement Policy iteration algorithm.

    Inputs:
    env: OpenAI Gym environment.
            env.P: dictionary
                    P[state][action] is list of tuples. Each tuple contains probability, nextstate, reward, terminal
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    gamma: float
            Discount factor.
    max_iteration: int
            The maximum number of iterations to run before stopping.
    theta: float
            The threshold of convergence.
            
    Outputs:
    V: numpy.ndarray
    policy: numpy.ndarray
    numIterations: int
    """

    V = np.zeros(env.nS)
    policy = np.zeros(env.nS, dtype = np.int32)
    policy_stable = False
    numIterations = 0
    
    while not policy_stable and numIterations < max_iteration:
        #Implement it with function policy_evaluation and policy_improvement
        ############################
        # YOUR CODE STARTS HERE
        
        # policy evaluation
        V = policy_evaluation(env, policy, gamma, theta)

        # update the policy
        policy, policy_stable = policy_improvement(env, V, policy, gamma)

        # YOUR CODE ENDS HERE
        ############################
        numIterations += 1
        
    return V, policy, numIterations


def policy_evaluation(env, policy, gamma, theta):
    """
    Evaluate the value function from a given policy.

    Inputs:
    env: OpenAI Gym environment.
            env.P: dictionary
                    
            env.nS: int
                    number of states
            env.nA: int
                    number of actions

    gamma: float
            Discount factor.
    policy: numpy.ndarray
            The policy to evaluate. Maps states to actions.
    theta: float
            The threshold of convergence.
    
    Outputs:
    V: numpy.ndarray
            The value function from the given policy.
    """
    ############################
    # YOUR CODE STARTS HERE

    nS = env.nS

    V_pi = np.zeros(nS)

    numIterations = 0

    delta = theta
    while delta>=theta and numIterations < 500:
    # while delta >= theta:
        delta = 0

        V = V_pi.copy()

        for s in range(nS):
            # because we use greedy strategy to choose policy for each state,
            # so each state only have one action with probability 1
            # policy is an array with shape (nS,), here the action is policy[s]
            q_sa_pi = 0
            for i in range(len(env.P[s][policy[s]])):
                # with action = policy[s], the state value = q(s,policy[s])
                prob, nextstate, reward, terminal = env.P[s][policy[s]][i]
                q_sa_pi += prob * (reward + gamma*V_pi[nextstate])
            V_pi[s] = q_sa_pi

            delta = max(delta, abs(V_pi[s]-V[s]))

        numIterations += 1
    # print(numIterations)
    
    # YOUR CODE ENDS HERE
    ############################

    return V


def policy_improvement(env, value_from_policy, policy, gamma):
    """
    Given the value function from policy, improve the policy.

    Inputs:
    env: OpenAI Gym environment
            env.P: dictionary
                    P[state][action] is tuples with (probability, nextstate, reward, terminal)
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
            env.nS: int
                    number of states
            env.nA: int
                    number of actions

    value_from_policy: numpy.ndarray
            The value calculated from the policy
    policy: numpy.ndarray
            The previous policy.
    gamma: float
            Discount factor.

    Outputs:
    new policy: numpy.ndarray
            An array of integers. Each integer is the optimal action to take
            in that state according to the environment dynamics and the
            given value function.
    policy_stable: boolean
            True if the "optimal" policy is found, otherwise false
    """
    ############################
    # YOUR CODE STARTS HERE

    policy_stable = False

    q = action_evaluation(env, gamma, value_from_policy)

    policy_new = np.argmax(q,axis=1)

    if policy_new is policy:
        policy_stable = True

    policy = policy_new

    # YOUR CODE ENDS HERE
    ############################

    return policy, policy_stable