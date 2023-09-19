# POLICY ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the policy iteration algorithm.

## PROBLEM STATEMENT
The bandit slippery walk problem is a reinforcement learning problem in which an agent must learn to navigate a 7-state environment in order to reach a goal state. The environment is slippery, so the agent has a chance of moving in the opposite direction of the action it takes.

### States
The environment has 7 states:

- Two Terminal States:
  * **G**: The goal state
  * **H**: A hole state.
- Five Transition states / Non-terminal States including
  * **S**: The starting state.
### Actions
The agent can take two actions:

* **R**: Move right.
* **L**: Move left.
### Transition Probabilities
The transition probabilities for each action are as follows:
- 50% chance that the agent moves in the intended direction.
- 33.33% chance that the agent stays in its current state.
- 16.66% chance that the agent moves in the opposite direction.
For example, if the agent is in state S and takes the "R" action, then there is a 50% chance that it will move to state 4, a 33.33% chance that it will stay in state S, and a 16.66% chance that it will move to state 2.

### Rewards
The agent receives a reward of +1 for reaching the goal state (G). The agent receives a reward of 0 for all other states.

### Graphical Representation
![image](https://github.com/Meenakshi0907/policy-iteration-algorithm/assets/94165108/8c677589-4a1d-4b57-acc8-407caea3acd4)

## POLICY ITERATION ALGORITHM
The algorithm implemented in the policy_iteration is a method used to find the optimal policy in a Markov decision process (MDP). Here's a step-by-step explanation of the algorithm:

Initialize the policy pi. In this implementation, a random action is chosen for each state s in the MDP P. The initial policy is represented by the lambda function pi=lambda s:{s:a for s,a in enumerate(random_actions)}[s], where random_actions is a list of randomly chosen actions for each state.

Enter a loop that continues until the policy pi is no longer changing. This is determined by comparing the previous policy (old_pi) with the current policy computed in the loop.

Store the previous policy as old_pi for comparison later.

Perform policy evaluation using the function policy_evaluation. This step calculates the state-values (V) for each state s given the current policy pi. The state-values represent the expected cumulative rewards starting from state s following policy pi and discounting future rewards by a factor of gamma. The function policy_evaluation is called with the arguments pi, P, gamma, and theta.

Perform policy improvement using the function policy_improvement. This step updates the policy pi based on the current state-values V. The function policy_improvement is called with the arguments V, P, and gamma.

Check if the policy has converged by comparing the previous policy old_pi with the current policy {s:pi(s) for s in range(len(P))}. If they are the same for all states s, the loop is exited.

Return the final state-values V and the optimal policy pi.

To summarize, policy iteration iteratively improves the policy by alternating between policy evaluation and policy improvement steps until convergence is reached. The algorithm guarantees to find the optimal policy for the given MDP P with a discount factor gamma.

## POLICY IMPROVEMENT FUNCTION
```py
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    # Write your code here to implement policy improvement algorithm
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob,next_state,reward,done in P[s][a]:
          Q[s][a]+=prob* (reward + gamma * V[next_state]*(not done))
          new_pi= lambda s: {s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return new_pi
```
## POLICY ITERATION FUNCTION
```py
def policy_iteration(P, gamma=1.0, theta=1e-10):
    random_actions = np.random.choice(tuple(P[0].keys()), len(P))
    # Write your code here to implement the policy iteration algorithm
    pi= lambda s:{s:a for s,a in enumerate(random_actions)}[s]
    while True:
      old_pi= {s:pi(s) for s in range(len(P))}
      V= policy_evaluation(pi,P,gamma,theta)
      pi=policy_improvement(V,P,gamma)
      if old_pi=={s:pi(s) for s in range(len(P))}:
        break
    return V, pi
```
## OUTPUT:
### POLICY IMPROVEMENT FUNCTION
![image](https://github.com/Meenakshi0907/policy-iteration-algorithm/assets/94165108/ac5613a1-51d0-4ca3-9667-b1bd2fb18611)
![image](https://github.com/Meenakshi0907/policy-iteration-algorithm/assets/94165108/ef86e36d-b7b1-4c96-a12e-6abca8b31d05)

### POLICY ITERATION FUNCTION
![image](https://github.com/Meenakshi0907/policy-iteration-algorithm/assets/94165108/0a3e5552-d3f4-4bf3-a031-ade54b849f36)
![image](https://github.com/Meenakshi0907/policy-iteration-algorithm/assets/94165108/d02f0847-1a38-4f4d-820e-4983e66b7d3c)

## RESULT:
Thus, a Python program is developed to find the optimal policy for the given MDP using the policy iteration algorithm.
