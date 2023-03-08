import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld
from algorithms import *
from plotting_utils import *

def main():
    env = discrete_pendulum.Pendulum(n_theta=15, n_thetadot=21)

    alpha_array = np.array([0.001, 0.01, 0.1, 0.5])
    eps_array = np.array([0.001, 0.01, 0.1])

    n_epochs = 10000

    for i in range(alpha_array.size):
        for j in range(eps_array.size):
            alpha = alpha_array[i]
            eps = eps_array[j]
            Q_s, pi_s, rtotal_s = SARSA(env, alpha, eps, n_epochs)
            Q_q, pi_q, rtotal_q = Q_off_policy(env, alpha, eps, n_epochs)

            plot_SARSA(rtotal_s, eps, alpha)
            plot_Qlearning(rtotal_q, eps, alpha)

    plot_example_traj_pend(env, pi_s, "Example Trajectory for SARSA: eps="+str(eps)+", alpha="+str(alpha))

    plot_policy(pi_s, "Policy from SARSA: eps="+str(eps)+", alpha="+str(alpha))

    plot_state_value(env, pi_s, alpha, n_epochs, "State Value Function from SARSA: eps="+str(eps)+", alpha="+str(alpha))

    plot_example_traj_pend(env, pi_q, "Example Trajectory for Q-learning: eps="+str(eps)+", alpha="+str(alpha))

    plot_policy(pi_q, "Policy from Q-learning: eps="+str(eps)+", alpha="+str(alpha))

    plot_state_value(env, pi_q, alpha, n_epochs, "State Value Function from Q-learning: eps="+str(eps)+", alpha="+str(alpha))


    plt.show()

if __name__ == '__main__':
    main()
