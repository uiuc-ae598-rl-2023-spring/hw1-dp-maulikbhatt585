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
    n_eval = 1000

    V_PI_mean = np.array(V_PI.mean())

    stable = False

    while(not stable):
        # for i in range(n_eval):
        while(True):
                delta = 0
                for j in range(n_states):
                    v = V_PI[j]
                    V_PI[j] = 0
                    a = pi[j]
                    for k in range(n_states):
                        V_PI[j] += env.p(k,j,a)*(env.r(j,a) + gamma*V_PI[k])

                    delta = np.maximum(delta,np.abs(v - V_PI[j]))
                V_PI_mean = np.append(V_PI_mean, V_PI.mean())
                if delta<1e-3:
                    print("Convergence happened!")
                    break

        stable = True

        for j in range(n_states):
                old_a = pi[j]

                values = np.zeros(n_actions)

                for a in range(n_actions):
                    for k in range(n_states):
                        values[a] += env.p(k,j,a)*(env.r(j,a) + gamma*V_PI[k])

                pi[j] = np.argmax(values)

                if pi[j] != old_a:
                    stable = False

    return V_PI, pi, V_PI_mean

def Value_Iteration(env):
    gamma = 0.95
    n_states = env.num_states
    n_actions = env.num_actions

    max_steps = env.max_num_steps

    V_VI = np.random.randn(n_states)

    V_VI_mean = np.array(V_VI.mean())

    pi = np.random.randint(4, size=n_states)

    while(True):
            delta = 0
            for j in range(n_states):
                v = V_VI[j]
                V_VI[j] = 0
                values = np.zeros(n_actions)
                for a in range(n_actions):
                    for k in range(n_states):
                        values[a] += env.p(k,j,a)*(env.r(j,a) + gamma*V_VI[k])

                V_VI[j] = np.max(values)

                delta = np.maximum(delta,np.abs(v - V_VI[j]))
            V_VI_mean = np.append(V_VI_mean, V_VI.mean())
            if delta<1e-3:
                print("Convergence happened!")
                break

    for j in range(n_states):
        values = np.zeros(n_actions)

        for a in range(n_actions):
            for k in range(n_states):
                values[a] += env.p(k,j,a)*(env.r(j,a) + gamma*V_VI[k])

        pi[j] = np.argmax(values)

    return V_VI, pi, V_VI_mean

def SARSA(env, alpha, eps, n_epochs):
    gamma = 0.95
    n_states = env.num_states
    n_actions = env.num_actions

    Q = np.random.randn(n_states, n_actions)

    num_epochs = n_epochs

    rtotal_s = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        s = env.reset()
        p = np.random.random()
        if p < eps:
            a = random.randrange(n_actions)
        else:
            a = np.argmax(Q[s,:])
        done = False
        t = 0
        while not done:
            s1, r, done = env.step(a)
            rtotal_s[epoch] += (gamma**t)*r
            t+=1

            p = np.random.random()
            if p < eps:
                a1 = random.randrange(n_actions)
            else:
                a1 = np.argmax(Q[s1,:])

            Q[s,a] = Q[s,a] + alpha*(r + gamma*Q[s1,a1] - Q[s,a])
            s = s1
            a = a1

    pi = np.array([np.argmax(Q[i,:]) for i in range(n_states)])

    return Q, pi, rtotal_s

def Q_off_policy(env, alpha, eps, n_epochs):
    gamma = 0.95
    n_states = env.num_states
    n_actions = env.num_actions

    Q = np.random.randn(n_states, n_actions)

    num_epochs = n_epochs

    rtotal_q = np.zeros(num_epochs)

    for epochs in range(num_epochs):
        s = env.reset()
        done = False
        t = 0
        while not done:
            p = np.random.random()
            if p < eps:
                a = random.randrange(n_actions)
            else:
                a = np.argmax(Q[s,:])
            s1, r, done = env.step(a)
            rtotal_q[epochs] += (gamma**t)*r
            t+=1
            Q[s,a] = Q[s,a] + alpha*(r + gamma*np.argmax(Q[s1,:]) - Q[s,a])
            s = s1

    pi = np.array([np.argmax(Q[i,:]) for i in range(n_states)])

    return Q, pi, rtotal_q

def TD_0(env, pi, alpha, n_epochs):
    gamma = 0.95
    n_states = env.num_states
    n_actions = env.num_actions
    V = np.random.randn(n_states)

    num_epochs = n_epochs

    for epochs in range(num_epochs):
        s = env.reset()
        done = False
        while not done:
            a = pi[s]
            s1, r, done = env.step(a)
            V[s] = V[s] + alpha*(r + gamma*V[s1] - V[s])
            s = s1
    return V
