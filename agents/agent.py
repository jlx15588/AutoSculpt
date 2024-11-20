import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical

from agents.common import PolicyNet, ValueNet, compute_advantage
from utils.pruning_algorithm import patterns


class Agent:
    def __init__(self):
        pass

    def take_action(self, *args):
        raise NotImplementedError

    def update(self, *args):
        raise NotImplementedError


class PPO(Agent):
    def __init__(self, num_in_feats, num_hidden_feats, num_embed_feats, num_hiddens, num_actions,
                 actor_lr, critic_lr, gamma, lmbda, agt_upt_epochs, eps, device):
        super(PPO, self).__init__()
        self.actor = PolicyNet(num_in_feats, num_hidden_feats, num_embed_feats, num_hiddens, num_actions, device)
        self.critic = ValueNet(num_in_feats, num_hidden_feats, num_embed_feats, device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.action_std = torch.full((1, num_actions), .2).to(device)

        self.gamma = gamma
        self.lmbda = lmbda
        self.num_epochs = agt_upt_epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        mu, std = self.actor(state)  # mu,std's shape: [1, 55]
        # covar_mat = torch.diag_embed(std * std)
        covar_mat = torch.diag_embed(self.action_std * self.action_std)
        action_dist = MultivariateNormal(mu, covar_mat)
        action = action_dist.sample()
        action = action.clamp(0.1, 0.9)
        return action  # [1, 55]

    def update(self, transition_dict):
        states = transition_dict['states']  # states: tuple[dict]
        actions = torch.tensor(transition_dict['actions']).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).view(-1, 1).to(self.device)
        next_states = transition_dict['next_states']
        dones = torch.tensor(transition_dict['dones'], dtype=torch.int8).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        states_values = self.critic(states)
        td_delta = td_target - states_values
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)

        old_mu, old_std = self.actor(states)
        old_covar_mat = torch.diag_embed(old_std * old_std)
        old_action_dists = MultivariateNormal(old_mu, old_covar_mat)
        old_log_probs = old_action_dists.log_prob(actions).view(-1, 1)

        for epoch in range(self.num_epochs):
            mu, std = self.actor(states)
            covar_mat = torch.diag_embed(std * std)
            action_dists = MultivariateNormal(mu, covar_mat)
            log_probs = action_dists.log_prob(actions).view(-1, 1)

            ratio = torch.exp(log_probs - old_log_probs.detach())
            surr1 = ratio * advantage.detach()
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage.detach()
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


class PPO2(Agent):
    def __init__(self, num_in_feats, num_hidden_feats, num_embed_feats, num_hiddens, num_actions,
                 actor_lr, critic_lr, gamma, lmbda, agt_upt_epochs, eps, device):
        """

        :param num_in_feats:
        :param num_hidden_feats:
        :param num_embed_feats:
        :param num_hiddens:
        :param num_actions:
        :param actor_lr:
        :param critic_lr:
        :param gamma:
        :param lmbda:
        :param agt_upt_epochs:
        :param eps:
        :param device:
        """
        super(PPO2, self).__init__()
        self.actor = PolicyNet(num_in_feats, num_hidden_feats, num_embed_feats, num_hiddens, num_actions, device)
        self.critic = ValueNet(num_in_feats, num_hidden_feats, num_embed_feats, device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # self.optimizer = torch.optim.Adam([
        #     {'params': self.actor.parameters(), 'lr': actor_lr},
        #     {'params': self.critic.parameters(), 'lr': critic_lr}
        # ])

        self.old_actor = PolicyNet(num_in_feats, num_hidden_feats, num_embed_feats, num_hiddens, num_actions, device)
        self.old_actor.load_state_dict(self.actor.state_dict())

        # action_std = torch.full((1, num_actions), .2).to(device)
        # self.covar_mat = torch.diag_embed(action_std * action_std)

        action_std = torch.full((1, len(patterns)), .2).to(device)
        self.covar_mat = torch.diag_embed(action_std * action_std)

        self.num_actions = num_actions
        self.gamma = gamma
        self.lmbda = lmbda
        self.num_epochs = agt_upt_epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        # mu, _ = self.old_actor(state)  # mu,std's shape: [1, 55]
        # action_dist = MultivariateNormal(mu, self.covar_mat)
        # action = action_dist.sample()
        # action = action.clamp(0.1, 0.9)
        # log_prob = action_dist.log_prob(action)
        # return action, log_prob  # [1, 55]

        # action, log_probs = self.old_actor(state)  # list[tensor], tensor shape: [1, 本层卷积核数]
        # return action, log_probs

        # probs = self.old_actor(state)
        mu = self.old_actor(state)
        cate_dist = MultivariateNormal(mu, self.covar_mat)
        probs = cate_dist.sample()  # [1, len(patterns)]
        probs = probs.clamp(0.01, 0.99)
        log_prob = cate_dist.log_prob(probs)

        action_dist = Categorical(probs)
        actions = []
        for num_filters in self.num_actions:
            action = action_dist.sample([num_filters])
            actions.append(action.T)
        return actions, probs, log_prob

    def update(self, transition_dict):
        states = transition_dict['states']  # states: tuple[dict]
        actions = torch.tensor(transition_dict['actions']).to(self.device)
        old_log_probs = torch.tensor(transition_dict['log_probs']).view(-1, 1).to(self.device)
        rewards = transition_dict['rewards']
        dones = transition_dict['dones']

        discounted_rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).view(-1, 1).to(self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        for epoch in range(self.num_epochs):
            mu, _ = self.actor(states)
            action_dists = MultivariateNormal(mu, self.covar_mat)
            log_probs = action_dists.log_prob(actions).view(-1, 1)
            dist_entropy = action_dists.entropy()

            states_values = self.critic(states)
            ratio = torch.exp(log_probs - old_log_probs.detach())
            advantages = discounted_rewards - states_values.detach()

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(states_values, discounted_rewards))
            # loss = torch.mean(-torch.min(surr1, surr2) + 0.5 * F.mse_loss(states_values, discounted_rewards) - 0.01 * dist_entropy)

            print(f'({actor_loss:.4f}, {critic_loss:.4f})', end=' ')
            # print(f'{loss:.4f},', end=' ')

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            # self.optimizer.zero_grad()
            # loss.mean().backward()
            # self.optimizer.step()

        self.old_actor.load_state_dict(self.actor.state_dict())

    def update2(self, transition_dict):
        states = transition_dict['states']  # states: tuple[graph]
        probs = torch.tensor(transition_dict['probs']).to(self.device)
        old_log_probs = torch.tensor(transition_dict['log_probs']).view(-1, 1).to(self.device)
        rewards = transition_dict['rewards']
        dones = transition_dict['dones']

        # old_log_probs = torch.log(self.actor(states).mean(dim=1, keepdim=True)).detach()

        discounted_rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).view(-1, 1).to(self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        for epoch in range(self.num_epochs):
            # _, log_probs = self.actor(states)
            # log_probs = torch.log(self.actor(states).mean(dim=1, keepdim=True))
            mu = self.actor(states)
            cate_dist = MultivariateNormal(mu, self.covar_mat)
            log_probs = cate_dist.log_prob(probs).view(-1, 1)

            states_values = self.critic(states)
            ratio = torch.exp(log_probs - old_log_probs.detach())
            advantages = discounted_rewards - states_values.detach()

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(states_values, discounted_rewards))

            print(f'({actor_loss:.4f}, {critic_loss:.4f})', end=' ')

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        self.old_actor.load_state_dict(self.actor.state_dict())
