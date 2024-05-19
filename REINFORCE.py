from torch.distributions import Categorical
import torch.optim as optim
import gymnasium as gym
import torch.nn as nn
import numpy as np
import torch
import kan


class Policy_Network(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super(Policy_Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class KolmogorovArnoldPolicyNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.kan = kan.KAN(width=[input_size, 5, 5, output_size], grid=5, k=5)
        self.output_activation = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_activation(self.kan(x))


class Agent:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        lr: float,
        gamma: float,
    ) -> None:
        self.policy_network = KolmogorovArnoldPolicyNetwork(input_size, output_size)
        self.optimizer = optim.RMSprop(self.policy_network.parameters(), lr)
        self.gamma = gamma
        self.log_prob_memory = []
        self.reward_memory = []

    def choose_action_train(self, state: np.ndarray) -> int:
        action_prob_dist = Categorical(
            self.policy_network.forward(
                torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)))
        action = action_prob_dist.sample()
        log_prob = action_prob_dist.log_prob(action)
        self.log_prob_memory.append(log_prob)
        return int(action.detach().item())

    def choose_action_test(self, state: np.ndarray) -> int:
        return int(self.policy_network.forward(
            torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        ).detach().argmax().item())

    def store_reward(self, reward: float) -> None:
        self.reward_memory.append(reward)

    def update(self) -> float:
        T = len(self.reward_memory)
        returns = torch.zeros(T, dtype=torch.float32)
        returns_sum = 0.0
        for i in range(T - 1, -1, -1):
            returns_sum = self.reward_memory[i] + self.gamma * returns_sum
            returns[i] = returns_sum
        returns = (returns - returns.mean()) / returns.std()
        loss = torch.tensor(0.0, dtype=torch.float32)
        self.optimizer.zero_grad()
        for return_, log_prob in zip(returns, self.log_prob_memory):
            loss -= return_ * log_prob[0]
        loss.backward()
        self.optimizer.step()

        self.log_prob_memory.clear()
        self.reward_memory.clear()

        return loss.item()

    def train(self, env: gym.Env, iteration: int) -> None:
        for i in range(iteration):
            state, _ = env.reset()
            score = 0.0
            done = False
            while not done:
                action = self.choose_action_train(state)
                state, reward, done, _, _ = env.step(action)
                reward = float(reward)
                self.store_reward(reward)
                score += reward
            loss = self.update()
            print(f"Iteration: {i + 1}, Score: {score}, Loss: {loss}")
            if i % 5 == 0:
                with open("./temp.pt", "wb") as f:
                    torch.save(self.policy_network.state_dict(), f)

    def test(self, env: gym.Env) -> None:
        state, _ = env.reset()
        score = 0.0
        done = False
        while not done:
            action = self.choose_action_train(state)
            state, reward, done, _, _ = env.step(action)
            reward = float(reward)
            score += reward



if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(
        4, 2, 0.001, 0.99
    )
    agent.train(env, 10000)

    # env = gym.make("CartPole-v1", render_mode="human")
    # agent = Agent(
    #     4, 2, 0.001, 0.99
    # )
    # with open("./temp.pt", "rb") as f:
    #     agent.policy_network.load_state_dict(torch.load(f))
    # agent.test(env)
