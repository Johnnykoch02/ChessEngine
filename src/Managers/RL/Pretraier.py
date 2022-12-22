from torch.utils.data.dataset import Dataset, random_split
from gzip import GzipFile
import numpy as np 
import torch as th
import gym
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import json
import matplotlib.pyplot as plt

from gym import spaces
from math import exp
from stable_baselines3 import PPO
# from custom_network import ppo_model, env
from gym.spaces import Box

class ExpertDataSet(Dataset):
    def __init__(self, expert_observations, expert_actions):
        self.observations = expert_observations
        self.actions = expert_actions
        self.keys = list(expert_observations.keys())
        
    def __getitem__(self, index):
        return ({key: self.observations[key][index] for key in self.keys}, self.actions[index])

    def __len__(self):
        return len(self.observations[self.keys[0]])

with GzipFile('formatted_actions_single.npy.gz', 'r') as f:
    expert_actions = np.load(f, allow_pickle = True)

    expert_actions = expert_actions[:, [1,2,3]]


with GzipFile('formatted_observations_single.npy.gz', 'r') as f:
    expert_observations = np.load(f, allow_pickle = True)
    print(x)


expert_observations = {
            # 'hand_config': expert_observations[:, 2],
            # 'hand_torque': expert_observations[:, 7],
            # 'finger_1_tactile': expert_observations[:, 4],
            # 'finger_2_tactile': expert_observations[:, 5],
            # 'finger_3_tactile': expert_observations[:, 6],
            # 'ball_count': expert_observations[:, 0],
            # 'ball_location': expert_observations[:, 1]  
            }

# distribution = {finger: {action:0 for action in range(3)} for finger in range(1,4)}
# for action in expert_actions:
  
#   distribution[1][action[0]] += 1
#   distribution[2][action[1]] += 1
#   distribution[3][action[2]] += 1

# print(distribution)


expert_dataset = ExpertDataSet(expert_observations, expert_actions)

train_size = int(0.8 * len(expert_dataset))

test_size = len(expert_dataset) - train_size

train_expert_dataset, test_expert_dataset = random_split(
    expert_dataset, [train_size, test_size]
)

print("test_expert_dataset: ", len(test_expert_dataset))
print("train_expert_dataset: ", len(train_expert_dataset))

# for i, j in test_expert_dataset:
#   for k, v in i.items():
#     print(v.shape)
#   break

# ppo_model.load("ppo_model_single")
# count = 0
# total = 0
# for i in test_expert_dataset:
#   x = ppo_model.predict(i[0])[0]
#   if all(x == i[1]):
#     count += 1
#   total += 1

# print('test', count, total, count/total)

# count = 0
# total = 0
# for i in train_expert_dataset:
#   x = ppo_model.predict(i[0])[0]
#   if all(x == i[1]):
#     count += 1
#   total += 1

# print('train', count, total, count/total)

# quit()

loss_series = []

def pretrain_agent(
    student,
    batch_size=128,
    epochs=1000,
    scheduler_gamma=0.7,
    learning_rate=1.0,
    log_interval=100,
    no_cuda=True,
    seed=1,
    test_batch_size=128,
):
    use_cuda = not no_cuda and th.cuda.is_available()
    # use_cuda = False
    th.manual_seed(seed)
    device = th.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    if isinstance(env.action_space, gym.spaces.Box):
      criterion = nn.MSELoss()
    else:
      criterion = nn.CrossEntropyLoss()
    
    # criterion = nn.MSELoss()

    # Extract initial policy
    model = student.policy.to(device)
    # model = student.policy

    def train(model, device, train_loader, optimizer):
        model.train(True)

        batch_losses = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data, target.to(device)
            # data, target = data, target
            
            for key, value in data.items():
              data[key] = data[key].to(device)
              # data[key] = data[key]
        
            optimizer.zero_grad()

            if isinstance(env.action_space, gym.spaces.Box):
              # A2C/PPO policy outputs actions, values, log_prob
              # SAC/TD3 policy outputs actions only
              if isinstance(student, (PPO)):
                action, _, _ = model(data)
              else:
                # SAC/TD3:
                action = model(data)
              action_prediction = action.double()
            else:
              # Retrieve the logits for A2C/PPO when using discrete actions
              dist = model.get_distribution(data)
              # action_prediction = [i.logits for i in dist.distribution]
              action_prediction = [i.probs for i in dist.distribution]
              # print(action_prediction)
              target = target.long()

            loss1 = criterion(action_prediction[0], target[:, 0])
            loss2 = criterion(action_prediction[1], target[:, 1])
            loss3 = criterion(action_prediction[2], target[:, 2])

            loss = loss1 + loss2 + loss3

            loss.backward()
            optimizer.step()

            batch_losses.append(loss)
        print('TRAIN', epoch, sum(batch_losses))
        
        loss_series.append(sum(batch_losses))  

    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        with th.no_grad():
            for data, target in test_loader:
                data, target = data, target.to(device)
                # data, target = data, target

                for key, value in data.items():
                  data[key] = data[key].to(device)
                  # data[key] = data[key]

                if isinstance(env.action_space, gym.spaces.Box):
                  # A2C/PPO policy outputs actions, values, log_prob
                  # SAC/TD3 policy outputs actions only
                  if isinstance(student, (A2C, PPO)):
                    action, _, _ = model(data)
                  else:
                    # SAC/TD3:
                    action = model(data)
                  action_prediction = action.double()
                else:
                  # Retrieve the logits for A2C/PPO when using discrete actions
                  dist = model.get_distribution(data)
                  # action_prediction = [i.logits for i in dist.distribution]
                  action_prediction = [i.probs for i in dist.distribution]
                  target = target.long()

                loss1 = criterion(action_prediction[0], target[:, 0])
                loss2 = criterion(action_prediction[1], target[:, 1])
                loss3 = criterion(action_prediction[2], target[:, 2])

                test_loss = loss1 + loss2 + loss3

        print(f"Test set: Average loss: {test_loss}")

    # Here, we use PyTorch `DataLoader` to our load previously created `ExpertDataset` for training
    # and testing

    train_loader = th.utils.data.DataLoader(
        dataset=train_expert_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )

    test_loader = th.utils.data.DataLoader(
        dataset=test_expert_dataset, batch_size=test_batch_size, shuffle=True, **kwargs,
    )

    # Define an Optimizer and a learning rate schedule.
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay = 0.1)
    scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

    # Now we are finally ready to train the policy model.
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer)
        test(model, device, test_loader)
        scheduler.step()

    # Implant the trained policy network back into the RL student agent
    ppo_model.policy = model


pretrain_agent(
    ppo_model,
    epochs=1000,
    scheduler_gamma=0.7,
    learning_rate= 3e-4,
    log_interval=100,
    no_cuda=False,
    seed=1,
    batch_size=128,
    test_batch_size=128,
)

ppo_model.save("ppo_model_single")

# plt.plot(list(range(len(loss_series))), loss_series)
# plt.savefig('loss.jpg')