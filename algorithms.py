import random
import numpy as np
import matplotlib.pyplot as plt
import gridworld
import discrete_pendulum

def Policy_Iteration(env):
    gamma = 0.95
    n_states = env.num_states
    n_actions = env.num_actions

    max_steps = env.max_num_steps

    V_PI = np.random.randn(n_states)

    pi = np.random.randint(4, size=n_states)

    n_eval = 500

    while(not stable):
        for i in range(n_eval):
                delta = 0
                for j in range(n_states):
                    v = V_PI[j]
                    V_PI[j] = 0
                    a = pi[j]
                    for k in range(n_states):
                        V_PI[j] += env.p(k,j,a)*env.r(j,a)*(env.r(j,a) + gamma*V_PI[k])

                    delta = np.maximum(delta,np.abs(v - V_PI[j]))
                if delta<1e-3:
                    print("Convergence happened!")
                    break

        stable = True

        for j in range(n_states):
                old_a = pi[j]

                values = np.zeros(n_actions)

                for a in range(n_actions):
                    for k in range(n_states):
                        values[a] += env.p(k,j,a)*env.r(j,a)*(env.r(j,a) + gamma*V_PI[k])

                pi[j] = np.argmax(values)

                if pi[j] != old_a:
                    stable = False

    return V_PI, pi

def Value_Iteration(env):
    gamma = 0.95
    n_states = env.num_states
    n_actions = env.num_actions

    max_steps = env.max_num_steps

    V_VI = np.random.randn(n_states)

    pi = np.random.randint(4, size=n_states)

    while(True):
            delta = 0
            for j in range(n_states):
                v = V_VI[j]
                V_VI[j] = 0
                values = np.zeros(n_actions)
                for a in range(n_actions):
                    for k in range(n_states):
                        values[a] += env.p(k,j,a)*env.r(j,a)*(env.r(j,a) + gamma*V_VI[k])

                V_VI[j] = np.max(values)

                delta = np.maximum(delta,np.abs(v - V_VI[j]))
            print(delta)
            if delta<1e-3:
                print("Convergence happened!")
                break

    for j in range(n_states):
        values = np.zeros(n_actions)

        for a in range(n_actions):
            for k in range(n_states):
                values[a] += env.p(k,j,a)*env.r(j,a)*(env.r(j,a) + gamma*V_VI[k])

        pi[j] = np.argmax(values)

    return V_VI, pi

def SARSA(env):
    alpha = 0.01
    eps = 1e-3
    gamma = 0.95

    Q = np.random.randn(n_states, n_actions)

    num_epochs = int(1e4)

    for epochs in range(num_epochs):
        s = env.reset()
        a = np.argmax(Q[s,:])
        done = False
        while not done:
            s1, r, done = env.step(a)
            a1 = np.argmax(Q[s1,:])
            Q[s,a] = Q[s,a] + alpha*(r + gamma*Q[s1,a1] - Q[s,a])
            s = s1
            a = a1

    pi = np.array([np.argmax(Q[i,:]) for i in range(n_states)])

    return Q, pi

def Q_off_policy(env):
    alpha = 0.01
    eps = 1e-3
    gamma = 0.95

    Q = np.random.randn(n_states, n_actions)

    num_epochs = int(1e4)

    for epochs in range(num_epochs):
        s = env.reset()
        done = False
        while not done:
            a = np.argmax(Q[s,:])
            s1, r, done = env.step(a)
            Q[s,a] = Q[s,a] + alpha*(r + gamma*np.argmax(Q[s1,:]) - Q[s,a])
            s = s1

    pi = np.array([np.argmax(Q[i,:]) for i in range(n_states)])

    return Q, pi

def TD_0(env, pi):
    alpha = 0.01
    V = np.random.randn(n_states)

    num_epochs = int(1e3)

    for epochs in range(num_epochs):
        s = env.reset()
        done = False
        while not done:
            a = pi[s]
            s1, r, done = env.step(a)
            V[s] = V[s] + alpha*(r + gamma*V[s1] - V[s])
            s = s1
    return V

# if __name__ == '__main__':
#
#     env = gridworld.GridWorld(hard_version=False)
#     s = env.reset
#
#     V_PI, pi, stable, delta = Policy_Iteration(env)
#
#     print(stable)
#     print(delta)
