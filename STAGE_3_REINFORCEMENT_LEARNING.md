# 第三阶段：强化学习流程

## 目标

第三阶段用于理解 agent 如何通过与环境交互学习策略，继续保持“一个 Python 文件对应一个完整流程”。

## 计划流程

| 顺序 | 流程 | 核心概念 |
|---|---|---|
| 1 | Multi-Armed Bandit | exploration、exploitation、epsilon-greedy |
| 2 | REINFORCE | policy gradient、return |
| 3 | DQN | replay buffer、target network、epsilon decay |
| 4 | Actor-Critic | policy network、value network、advantage |
| 5 | PPO | clipped objective、GAE、multiple updates |
| 6 | SAC | continuous action、entropy、double Q network |

建议使用 Gymnasium 的经典小环境，例如 CartPole、LunarLander 和 Pendulum。

## 每个流程必须包含

```text
environment
-> state/action space
-> agent/network
-> collect experience
-> calculate return or target
-> update model
-> evaluation
-> save/load
-> inference/render
```

## 期望达到的效果

完成第三阶段后，应当能够：

- 区分 supervised learning 与 reinforcement learning
- 解释 state、action、reward、return、value 和 policy
- 独立实现 replay buffer 和 rollout buffer
- 解释 on-policy 与 off-policy
- 保存并加载 agent
- 绘制 episode reward 与训练稳定性曲线

论文级复现继续放在独立的 `paper-reproduction` 仓库中。
