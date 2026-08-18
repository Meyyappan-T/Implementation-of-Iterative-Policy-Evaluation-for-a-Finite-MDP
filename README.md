# Exp-2 Implementation of Iterative Policy Evaluation for a Finite-MDP
## Aim

To implement iterative policy evaluation using Gymnasium and estimate the state-value function $V^\pi(s)$ for a fixed random policy.


## Software Requirements

Install the required Python packages:

```bash
pip install gymnasium numpy
```



## Environment Used

The experiment uses the **FrozenLake-v1** environment from Gymnasium.

FrozenLake is a grid-based reinforcement learning environment where the agent starts from a start state and tries to reach the goal state without falling into holes.

For the default 4 x 4 FrozenLake map:

| Component | Description |
|---|---|
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching goal, 0 otherwise |
| Terminal states | Goal and holes |


## Problem Statement

Evaluate a fixed random policy in the FrozenLake-v1 environment.

The agent follows a random policy, where each of the four actions is selected with equal probability:

$$
\pi(a|s) = \frac{1}{4}
$$

This probability refers to the policy's action-selection probability. The environment transition probabilities are obtained from Gymnasium using `env.P[state][action]`. If `is_slippery=True`, the agent may not move in the intended direction due to stochastic transitions.

The objective is to estimate the state-value function:

$$
V^\pi(s)
$$



## Theory

The state-value function under policy $pi$, denoted by $V^\pi(s)$, represents the expected return starting from state $s$ and following policy $pi$.

The Bellman expectation equation is:

```math
V^\pi(s) =
\sum_a \pi(a|s)
\sum_{s'} P(s'|s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
```

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $\pi(a \mid s)$ | Probability of selecting action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $\gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $\pi$ |


## Algorithm

1. Create the FrozenLake-v1 environment using Gymnasium.
2. Access the transition model of the environment.
3. Initialize \(V(s)=0\) for all states.
4. Define a random policy where each action has equal probability.
5. For each state:
   - For each action:
     - Read transition probability, next state, reward, and terminal status.
     - Apply the Bellman expectation equation.
6. Repeat until the value function converges.
7. Display the final value function as a 4 x 4 grid.



## Program
### is_slippery=True
```python
import gymnasium as gym
import numpy as np
```

```python
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
```

```python
env = env.unwrapped
env.P
```

```python
n_states = env.observation_space.n
n_actions = env.action_space.n
```

```python
gamma = 0.99
theta = 1e-8
```

```python
policy = np.ones((n_states, n_actions)) / n_actions
```

```python
V = np.zeros(n_states)

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    V = np.zeros(n_states)
    iteration = 0

    while True:
        delta = 0

        for s in range(n_states):
            v = V[s]
            new_v = 0

            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    new_v += action_prob * prob * (
                        reward + gamma * V[next_state] * (not done)
                    )

            V[s] = new_v
            delta = max(delta, abs(v - V[s]))

        iteration += 1

        if delta < theta:
            break

    return V, iteration
```

```python
V, iterations = policy_evaluation(env, policy, gamma, theta)

print("Name: Meyyappan T")
print("Register Number: 212223240086")
print("Number of iterations:", iterations)

print("\nState-Value Function:")
print(V)

print("\nState-Value Function as 4x4 Grid:")
print(np.round(V.reshape(4, 4), 4))

env.close()
```

### is_slippery=False
```python
import gymnasium as gym
import numpy as np
```

```python
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
```

```python
env = env.unwrapped
env.P
```

```python
n_states = env.observation_space.n
n_actions = env.action_space.n
```

```python
gamma = 0.99
theta = 1e-8
```

```python
policy = np.ones((n_states, n_actions)) / n_actions
```

```python
V = np.zeros(n_states)

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    V = np.zeros(n_states)
    iteration = 0

    while True:
        delta = 0

        for s in range(n_states):
            v = V[s]
            new_v = 0

            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    new_v += action_prob * prob * (
                        reward + gamma * V[next_state] * (not done)
                    )

            V[s] = new_v
            delta = max(delta, abs(v - V[s]))

        iteration += 1

        if delta < theta:
            break

    return V, iteration
```

```python
V, iterations = policy_evaluation(env, policy, gamma, theta)

print("Name: Meyyappan T")
print("Register Number: 212223240086")
print("Number of iterations:", iterations)

print("\nState-Value Function:")
print(V)

print("\nState-Value Function as 4x4 Grid:")
print(np.round(V.reshape(4, 4), 4))

env.close()
```


## Output

### is_slippery=True
<img width="606" height="312" alt="image" src="https://github.com/user-attachments/assets/77dfa47a-9b79-4c64-a077-167a268e5cb0" />




### is_slippery=False

<img width="651" height="305" alt="image" src="https://github.com/user-attachments/assets/7036e116-90e9-4131-a9e7-8a2c5d545808" />


## Result

Iterative policy evaluation was implemented successfully using the Gymnasium FrozenLake environment. The state-value function for the fixed random policy was estimated using the Bellman expectation equation.



## Inference

```text
Both notebooks implement the same Iterative Policy Evaluation algorithm with identical code except for the is_slippery parameter.
is_slippery=False creates a deterministic environment where the agent always moves in the intended direction.
is_slippery=True creates a stochastic environment where the agent may slip, introducing randomness in state transitions.
The deterministic environment produces higher state-value estimates and converges faster than the stochastic environment.
The experiment demonstrates that changes in environment dynamics significantly affect the evaluated state-value function, even when the policy remains unchanged.
```
