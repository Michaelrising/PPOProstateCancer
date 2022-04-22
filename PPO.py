import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical






################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init, device):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.device = device

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(self.device)

        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 128),
                            nn.Tanh(),
                            nn.Linear(128, 128),
                            nn.Tanh(),
                            nn.Linear(128, 64),
                            nn.LeakyReLU(),
                            nn.Linear(64, action_dim)
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 128),
                            nn.Tanh(),
                            nn.Linear(128, 64),
                            nn.LeakyReLU(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )

        
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 128),
                        nn.Tanh(),
                        nn.Linear(128, 64),
                        nn.LeakyReLU(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()

    def act_exploit(self, state):
        with torch.no_grad():
            action_probs = self.actor(state)
            greedy_action = torch.argmax(
                action_probs.unsqueeze(0), dim=1, keepdim=False)
            dist = Categorical(action_probs)
            action_logprob = dist.log_prob(greedy_action.squeeze(-1))
        return greedy_action.detach(), action_logprob.detach()

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, num_env, device, decay_step_size = 1000,decay_ratio = 0.5, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        
        self.buffer = RolloutBuffer()
        self.buffers = [RolloutBuffer() for _ in range(num_env)]

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, self.device).to(self.device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=decay_step_size,
                                                         gamma=decay_ratio)

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, self.device).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob = self.policy_old.act(state)

            return state, action.item(), action_logprob

    def greedy_select_action(self, state):
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob = self.policy_old.act_exploit(state)

            return state, action.item(), action_logprob

    def update(self, decayflag, grad_clamp=None):
        rewards_all_envs = []
        old_states_all_envs =[]
        old_actions_all_envs = []
        old_logprobs_all_envs = []
        for i in range(len(self.buffers)):
            # Monte Carlo estimate of returns
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(self.buffers[i].rewards), reversed(self.buffers[i].is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            
            # Normalizing the rewards
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

            # convert list to tensor
            old_states = torch.squeeze(torch.stack(self.buffers[i].states, dim=0)).detach().to(self.device)
            old_actions = torch.squeeze(torch.tensor(self.buffers[i].actions)).detach().to(self.device)
            old_logprobs = torch.squeeze(torch.stack(self.buffers[i].logprobs, dim=0)).detach().to(self.device)

            rewards_all_envs.append(rewards)
            old_states_all_envs.append(old_states)
            old_actions_all_envs.append(old_actions)
            old_logprobs_all_envs.append(old_logprobs)


        # Optimize policy for K epochs
        VLoss = 0
        for _ in range(self.K_epochs):
            #for index in BatchSampler(SubsetRandomSampler(range(self.mini_batch), self.batch_size, True)):
            loss_sum = 0
            vloss_sum = 0
            for i in range(len(self.buffers)):
                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states_all_envs[i], old_actions_all_envs[i])

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs_all_envs[i].detach())

                # Finding Surrogate Loss
                advantages = rewards_all_envs[i] - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

                # final loss of clipped objective PPO
                v_loss = self.MseLoss(state_values, rewards_all_envs[i])
                loss = -torch.min(surr1, surr2) + 0.5*v_loss - 0.01*dist_entropy
                loss_sum += loss.mean()
                vloss_sum += v_loss.mean()
            
            # take gradient step
            self.optimizer.zero_grad()
            loss_sum.backward()
            if grad_clamp is not None:
                nn.utils.clip_grad_norm_(self.policy.parameters(), grad_clamp)
            self.optimizer.step()
            VLoss += vloss_sum
        if decayflag:
            self.scheduler.step()
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        for i in range(len(self.buffers)):
            self.buffers[i].clear()
        return VLoss.cpu().detach()/self.K_epochs
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       


