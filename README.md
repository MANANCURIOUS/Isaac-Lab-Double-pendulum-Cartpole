# Isaac-Lab-Double-pendulum-Cartpole
Task 2
# Double Cart Pendulum
This project extends the single Cartpole experiment to a Double Cart Double Pendulum system using Proximal Policy Optimization (PPO) implemented with SKRL (Torch backend).

The objective is to stabilize two coupled pendulum links mounted on a cart under randomized initial conditions.

## Problem Description
The environment:

```
Isaac-Cart-Double-Pendulum-Direct-v0
```
is a multi-agent direct reinforcement learning environment.

However, for this experiment we use single-policy PPO, where:
- A single neural network controls:
  - Cart force
  - Second joint torque
- Multi-agent observations are merged into a centralized control setup.

<img width="651" height="378" alt="image" src="https://github.com/user-attachments/assets/9347e7bf-bd12-4146-aa42-ed4d2f23301f" />

## Baseline PPO Configuration
```
agent:
  rollouts: 16
  learning_epochs: 8
  mini_batches: 1
  learning_rate: 3e-4
  entropy_loss_scale: 0.0
  discount_factor: 0.99
  lambda: 0.95

models:
  policy layers: [32, 32]
  value layers: [32, 32]

trainer:
  timesteps: 4800

```
### Issues With Baseline
- Rollouts too small → noisy gradient updates
- Only 1 mini-batch → unstable optimization
- No entropy → weak exploration
- Very small network → underfitting
- 4800 timesteps → insufficient training

Result:
- Slow convergence
- High oscillation
- Poor stabilization
### Improved PPO Configuration
To improve performance, the following changes were applied:
```
agent:
  rollouts: 64
  learning_epochs: 10
  mini_batches: 4
  learning_rate: 3e-4
  entropy_loss_scale: 0.01
  discount_factor: 0.99
  lambda: 0.95

models:
  policy layers: [128, 128]
  value layers: [128, 128]

trainer:
  timesteps: 200000

```

<img width="682" height="375" alt="image" src="https://github.com/user-attachments/assets/75e9f73b-4c80-4130-86a1-c79a536fe639" />

### Training Command
```
python scripts\reinforcement_learning\skrl\train.py --task Isaac-Cart-Double-Pendulum-Direct-v0 agent.agent.rollouts=64 agent.agent.learning_epochs=10 agent.agent.mini_batches=4 agent.agent.entropy_loss_scale=0.01 agent.trainer.timesteps=200000
```
### Observed Improvements
After tuning:
- Faster convergence
- Reduced oscillation
- Higher episode length mean
- More stable training curve
- Better coordination between cart and pendulum
### Video Results
## PPO 
### Initial Training Instance
https://github.com/user-attachments/assets/04dad9b8-c41f-4683-90e2-8fdce33f78d2

### Final Training Instance 
https://github.com/user-attachments/assets/05709fdf-c586-4687-b1a4-ab51ab631c6e

### Trained Model

https://github.com/user-attachments/assets/75e6b496-d545-488c-adae-f57d96ff8710

## Using MPPO Algorithm (Multi-Agent PPO)
MAPPO is an extension of PPO for multi-agent systems, where:

- Each agent has its own observation
- Policies are learned jointly
- Agents cooperate to maximize shared reward

### Training Command
```
python scripts\reinforcement_learning\skrl\train.py --task Isaac-Cart-Double-Pendulum-Direct-v0 --algorithm MAPPO --num_envs 64
```
### Play / Evaluate Trained Model
```
python scripts\reinforcement_learning\skrl\play.py --task Isaac-Cart-Double-Pendulum-Direct-v0 --algorithm MAPPO --num_envs 1
```

Baseline Hyperparameters
```
agent:
  rollouts: 16
  learning_epochs: 8
  mini_batches: 1
  learning_rate: 3e-4
  discount_factor: 0.99
  lambda: 0.95
  ratio_clip: 0.2
  value_clip: 0.2
  entropy_loss_scale: 0.0
```

Improvements Applied

To improve stability and learning:

- rollouts: 128
- learning_epochs: 10
- mini_batches: 8
- learning_rate: 5e-4
- network layers: [32, 32]
- entropy_loss_scale: 0.02

Observed Improvements:

- Faster convergence
- Reduced oscillations
- More stable double-pole balancing

Better cooperative behavior between agent

# Graph Result
<img width="1158" height="601" alt="image" src="https://github.com/user-attachments/assets/695ec1df-0b89-4877-9b7a-2014525d1800" />

<img width="1157" height="589" alt="Screenshot 2026-02-19 060816" src="https://github.com/user-attachments/assets/70562420-46d8-4c32-ab46-6cdc4e2c1a87" />

# Video Results
## MPPO
### Training Instance first
https://github.com/user-attachments/assets/8e968c03-a829-4bc1-b0c5-b7cee20e20a1

### Training Instance Final
https://github.com/user-attachments/assets/d621f0d2-ec30-490b-b16c-3e3417507814
### Baseline Model


https://github.com/user-attachments/assets/977e4020-0228-4de0-8016-0f866e63c9ae


### Tuned Model

https://github.com/user-attachments/assets/618026bb-e41b-446a-99b2-a6a49177dba2



