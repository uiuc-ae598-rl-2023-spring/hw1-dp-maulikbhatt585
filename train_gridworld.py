import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld
from algorithms import *
from plotting_utils import *

def main():
    env = gridworld.GridWorld(hard_version=False)

    V_PI, pi_PI, V_PI_mean = Policy_Iteration(env)

    print("Training via Policy Iteraton done!")

    plot_V_PI(V_PI_mean)

    plot_example_traj_grid(env, pi_PI, "Example Trajectory for Policy Iteration")

    plot_policy(pi_PI, "Policy from Policy Iteration")

    print("Starting Value Iteration")

    V_VI, pi_VI, V_VI_mean = Value_Iteration(env)

    print("Training via Value Iteraton done!")

    plot_V_VI(V_VI_mean)

    plot_example_traj_grid(env, pi_VI, "Example Trajectory for Value Iteration")

    plot_policy(pi_VI, "Policy from Value Iteration")

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

    plot_example_traj_grid(env, pi_s, "Example Trajectory for SARSA: eps="+str(eps)+", alpha="+str(alpha))

    plot_policy(pi_s, "Policy from SARSA: eps="+str(eps)+", alpha="+str(alpha))

    plot_state_value(env, pi_s, alpha, n_epochs, "State Value Function from SARSA: eps="+str(eps)+", alpha="+str(alpha))

    plot_example_traj_grid(env, pi_q, "Example Trajectory for Q-learning: eps="+str(eps)+", alpha="+str(alpha))

    plot_policy(pi_q, "Policy from Q-learning: eps="+str(eps)+", alpha="+str(alpha))

    plot_state_value(env, pi_q, alpha, n_epochs, "State Value Function from Q-learning: eps="+str(eps)+", alpha="+str(alpha))


    plt.show()

if __name__ == '__main__':
    main()
