import itertools
import matplotlib
import numpy as np
import sys

from itertools import product
from collections import defaultdict

import plotting #from lib 
import  matplotlib

matplotlib.style.use('ggplot')

""" Code lifted and adapted from https://github.com/dennybritz/reinforcement-learning
"""

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action. Float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA #. set probability for other actions as epsilon/nA 
        best_action = np.argmax(Q[observation]) #. observation is a 4-tuple 
        A[best_action] += (1.0 - epsilon) #. set prob for best action as 1-epsilon
        return A
    return policy_fn

def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon_range=(0.98, 0.1), verbosity=2):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance to sample a random action. Float between 0 and 1.
    
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    # Q = defaultdict(lambda: np.ones(env.action_space.n)*sys.maxsize)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))    
    
    # make a list of epsilons linearly divided to the number of episodes
    epsilons = np.linspace(epsilon_range[0], epsilon_range[1], num_episodes)
    for i_episode in range(num_episodes):

        # The policy we're following
        epsilon = epsilons[i_episode]
        policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
        
        # Reset the environment and pick the first action
        state = env.reset()
        
        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():
            
            # Take a step
            #. follow epsilon-greedy policy (behaviour policy mu)
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs) #.action here is an index
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            
            # TD Update
            #. choose the best next action according to current Q-values of the next state
            #. update our target policy greedy pi 
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
                
            if done:
                break
                
            state = next_state
    
        # Print out which episode results
        if (i_episode+1) % 100 == 0:
            if verbosity==1:
                print("\rEpisode {} ".format(i_episode+1), end="")
                sys.stdout.flush()
            elif verbosity==2:                
                print(f"\rEpisode {i_episode+1}/{num_episodes}: Cost {stats.episode_rewards[i_episode]} epsilon {epsilon:.2f}",end="")
                sys.stdout.flush()

    return Q, stats


def ql_get_policy(env, Q):
    """    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        Q is the optimal action-value function, a dictionary mapping state -> action values.
    
    Returns:
        policy: A dictionary containing list of actions per agent
    Note:
        This is not in the dennybritz github repo
    """

    action_space_tuples = tuple(product((0,1,2,3), repeat=4))

    policy = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
    }

    # Reset the environment and pick the first action
    state = env.reset()
    for t in itertools.count():   

        # Take a step following the policy in Q-Table
        action = np.argmax(Q[state])

        #convert action (integer 1..256) to a 4-tuple
        action_tuple = action_space_tuples[action]
        policy[1].append(action_tuple[0])
        policy[2].append(action_tuple[1])
        policy[3].append(action_tuple[2])
        policy[4].append(action_tuple[3])
        policy[5].append(0)

        # Go to the next state
        state, reward, done, _ = env.step(action)

        if done:
            break

    return policy