import random
import numpy as np
import matplotlib.pyplot as plt
from algorithms import *

def plot_V_PI(V_PI_mean):
    plt.figure()
    plt.plot(V_PI_mean)
    plt.title("Policy Iteration - Mean Value Function")
    plt.xlabel("Iterations")
    plt.ylabel("Value")

def plot_V_VI(V_VI_mean):
    plt.figure()
    plt.plot(V_VI_mean)
    plt.title("Value Iteration - Mean Value Function")
    plt.xlabel("Iterations")
    plt.ylabel("Value")

def plot_SARSA(rtotal_s, eps, alpha):
    plt.figure()
    plt.plot(rtotal_s)
    plt.xlabel("Iterations")
    plt.ylabel("Total Return")
    plt.title("SARSA, eps="+str(eps)+", alpha="+str(alpha))
    # plt.show()

def plot_Qlearning(rtotal_q, eps, alpha):
    plt.figure()
    plt.plot(rtotal_q)
    plt.xlabel("Iterations")
    plt.ylabel("Total Return")
    plt.title("Q-learning(Off-policy), eps="+str(eps)+", alpha="+str(alpha))
    # plt.show()

def plot_example_traj_grid(env, pi, title):
    s = env.reset()

    # Create log to store data from simulation
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }

    # Simulate until episode is done
    done = False
    while not done:
        a = pi[s]
        (s, r, done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)

    # Plot data and save to png file
    plt.figure()
    plt.plot(log['t'], log['s'])
    plt.plot(log['t'][:-1], log['a'])
    plt.plot(log['t'][:-1], log['r'])
    plt.legend(['s', 'a', 'r'])
    plt.title(title)
    # plt.show()
    #plt.savefig('figures/gridworld/test_gridworld.png')


def plot_example_traj_pend(env, pi, title):
    s = env.reset()

    # Create log to store data from simulation
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
        'theta': [env.x[0]],        # agent does not have access to this, but helpful for display
        'thetadot': [env.x[1]],     # agent does not have access to this, but helpful for display
    }

    # Simulate until episode is done
    done = False
    while not done:
        a = pi[s]
        (s, r, done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)
        log['theta'].append(env.x[0])
        log['thetadot'].append(env.x[1])

    # Plot data and save to png file
    plt.figure()
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(log['t'], log['s'])
    ax[0].plot(log['t'][:-1], log['a'])
    ax[0].plot(log['t'][:-1], log['r'])
    ax[0].legend(['s', 'a', 'r'])
    ax[1].plot(log['t'], log['theta'])
    ax[1].plot(log['t'], log['thetadot'])
    ax[1].legend(['theta', 'thetadot'])
    plt.title(title)
    # plt.show()



def plot_policy(pi, title):
    plt.figure()
    plt.plot(pi)
    plt.xlabel("State")
    plt.ylabel("Action")
    plt.title(title)
    # plt.show()


def plot_state_value(env, pi, alpha, n_epochs, title):
    V = TD_0(env, pi, alpha, n_epochs)
    plt.figure()
    plt.plot(V)
    plt.xlabel("State")
    plt.ylabel("Value")
    plt.title(title)
