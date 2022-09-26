import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import original_buffer
import PER_buffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Architecture from experimental details section of the DDPG 
#paper "Continuous control with deep reinforcement learning"
#Lillicrap et. al. 2015
class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, a_max):
        super(Actor, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a_max = a_max
        
        self.l1 = nn.Linear(self.s_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, self.a_dim)
  
    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x)) * self.a_max  
        return x 

#Architecture from experimental details section of the DDPG
#paper "Continuous control with deep reinforcement learning"
#Lillicrap et. al. 2015
class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Critic, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.l1 = nn.Linear(self.s_dim, 400)
        self.l2 = nn.Linear(400 + self.a_dim, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, s, a):
        x = F.relu(self.l1(s))
        x = torch.cat([x, a], 1)
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x 

class DDPG(object):
    def __init__(self, s_dim, a_dim, a_max):
        #Create actor and actor target
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.actor = Actor(s_dim, a_dim, a_max).to(device)
        self.actor_target = Actor(s_dim, a_dim, a_max).to(device)
        #Initialize actor and actor target exactly the same
        self.actor_target.load_state_dict(self.actor.state_dict())
        #Adam optimizer to train actor
        #Learning rate specified in DDPG paper
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        
        #Create critic and critic target
        self.critic = Critic(s_dim, a_dim).to(device)
        self.critic_target = Critic(s_dim, a_dim).to(device)
        #Initialize critic and critic target exactly the same
        self.critic_target.load_state_dict(self.critic.state_dict())
        #Adam optimizer to train critic
        #L2 weight decay specified in DDPG paper
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)
    
    #Given a state, the actor returns a policy 
    def get_action(self, s):
        s = torch.FloatTensor(s.reshape(1, -1)).to(device)
        return self.actor(s).cpu().data.numpy().flatten()

    def modify_policy(self, s_dim, a_dim, a_max, case):
        # Hopper -> Walker
        self.critic.a_dim = a_dim
        self.critic.s_dim = s_dim
        self.critic_target.a_dim = a_dim
        self.critic_target.s_dim = s_dim
        self.actor.a_dim = a_dim
        self.actor.s_dim = s_dim
        self.actor_target.a_dim = a_dim
        self.actor_target.s_dim = s_dim

        with torch.no_grad():
            if case == 0:
                # print(self.critic.l2.weight.shape)
                self.critic.l1.weight = nn.Parameter(torch.cat((self.critic.l1.weight, torch.zeros(400, 7)), 1))
                self.critic.l2.weight = nn.Parameter(torch.cat((self.critic.l2.weight, torch.zeros(300, 3)), 1))
                self.critic_target.l1.weight = nn.Parameter(torch.cat((self.critic_target.l1.weight, torch.zeros(400, 7))
                                                                      , 1))
                self.critic_target.l2.weight = nn.Parameter(torch.cat((self.critic_target.l2.weight, torch.zeros(300, 3))
                                                                      , 1))
                self.actor.l1.weight = nn.Parameter(torch.cat((self.actor.l1.weight, torch.zeros(400, 7)), 1))
                self.actor_target.l1.weight = nn.Parameter(torch.cat((self.actor_target.l1.weight, torch.zeros(400, 7)), 1))

                # print(self.actor.l3.bias)
                a_bias = nn.Parameter(torch.cat((self.actor.l3.bias, torch.zeros(3)), 0))
                a_weight = nn.Parameter(torch.cat([self.actor.l3.weight, torch.zeros(3, 300)]))
                self.actor.l3 = nn.Linear(300, a_dim)
                self.actor.l3.bias = a_bias
                self.actor.l3.weight = a_weight

                a_tag_bias = nn.Parameter(torch.cat((self.actor_target.l3.bias, torch.zeros(3)), 0))
                a_tag_weight = nn.Parameter(torch.cat([self.actor_target.l3.weight, torch.zeros(3, 300)]))
                self.actor_target.l3 = nn.Linear(300, a_dim)
                self.actor_target.l3.bias = a_tag_bias
                self.actor_target.l3.weight = a_tag_weight
                print(self.actor_target.l3.bias.shape)
                print(self.actor_target.l3.weight.shape)

            if case == 1:
                self.critic.l1.weight = nn.Parameter(self.critic.l1.weight[:, :s_dim])
                self.critic.l2.weight = nn.Parameter(self.critic.l2.weight[:, :403])
                self.critic_target.l1.weight = nn.Parameter(self.critic_target.l1.weight[:, :s_dim])
                self.critic_target.l2.weight = nn.Parameter(self.critic_target.l2.weight[:, :403])

                self.actor.l1.weight = nn.Parameter(self.actor.l1.weight[:, :s_dim])
                self.actor_target.l1.weight = nn.Parameter(self.actor_target.l1.weight[:, :s_dim])

                a_bias = nn.Parameter(self.actor.l3.bias[:3])
                a_weight = nn.Parameter(self.actor.l3.weight[:3])
                self.actor.l3 = nn.Linear(300, a_dim)
                self.actor.l3.weight = a_weight
                self.actor.l3.bias = a_bias

                a_tag_bias = nn.Parameter(self.actor_target.l3.bias[:3])
                a_tag_weight = nn.Parameter(self.actor_target.l3.weight[:3])
                self.actor_target.l3 = nn.Linear(300, a_dim)
                self.actor_target.l3.weight = a_tag_weight
                self.actor_target.l3.bias = a_tag_bias

            self.actor.a_max = a_max
            self.actor_target.a_max = a_max
            self.critic.to(device)
            self.actor.to(device)
            self.critic_target.to(device)
            self.actor_target.to(device)
            # print(self.actor_target.load_state_dict())
            #
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

    #Update actor, critic and target networks with minibatch of experiences
    def train(self, replay_buffer, prioritized, beta_value, epsilon, T, batch_size=64, gamma=0.99, tau=0.005):

        for i in range(T):
            # Sample replay buffer
            if prioritized: 
                #Prioritized experience replay
                experience = replay_buffer.sample(batch_size, beta_value)
                s, a, r, s_new, done, weights, batch_idxes = experience
                #reshape hopper_data
                r = r.reshape(-1, 1)
                # print("Actions: ", len(a[0]))
                # print("State: ", len(s[0]))
                # print("Reward: ", len(r[0]))
                done = done.reshape(-1, 1)
                #We do not use importance sampling weights
                #Therefore importance sampling weights are all set to 1
                #See Hyperparameter search in report 
                weights = np.ones_like(r)
                #weights = weights.reshape(-1, 1)
            else:
                #Uniform experience replay
                s, a, r, s_new, done = replay_buffer.sample(batch_size)
                #importance sampling weights are all set to 1
                weights, batch_idxes = np.ones_like(r), None

            #Sqrt weights 
            #We do this since each weight will squared in MSE loss
            weights = np.sqrt(weights)

            #convert hopper_data to tensors
            state = torch.FloatTensor(s).to(device)
            action = torch.FloatTensor(a).to(device)
            next_state = torch.FloatTensor(s_new).to(device)
            done = torch.FloatTensor(1 - done).to(device)
            reward = torch.FloatTensor(r).to(device)
            weights = torch.FloatTensor(weights).to(device)
   
            #Compute the Q value estimate of the target network
            Q_target = self.critic_target(next_state, self.actor_target(next_state))
            #Compute Y
            Y = reward + (done * gamma * Q_target).detach()
            #Compute Q value estimate of critic
            Q = self.critic(state, action)
            #Calculate TD errors
            TD_errors = (Y - Q)
            #Weight TD errors 
            weighted_TD_errors = torch.mul(TD_errors, weights)
            #Create a zero tensor
            zero_tensor = torch.zeros(weighted_TD_errors.shape)
            #Compute critic loss, MSE of weighted TD_r
            critic_loss = F.mse_loss(weighted_TD_errors,zero_tensor)

            #Update critic by minimizing the loss
            #https://pytorch.org/docs/stable/optim.html
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            #Update the actor policy using the sampled policy gradient:
            #https://pytorch.org/docs/stable/optim.html
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the target models
            for critic_weights, critic__target_weights in zip(self.critic.parameters(), self.critic_target.parameters()):
                critic__target_weights.data.copy_(tau * critic_weights.data + (1 - tau) * critic__target_weights.data)
            for actor_weights, actor__target_weights in zip(self.actor.parameters(), self.actor_target.parameters()):
                actor__target_weights.data.copy_(tau * actor_weights.data + (1 - tau) * actor__target_weights.data)
    
            #For prioritized exprience replay
            #Update priorities of experiences with TD errors
            if prioritized:
                td_errors = TD_errors.detach().numpy()
                new_priorities = np.abs(td_errors) + epsilon
                replay_buffer.update_priorities(batch_idxes, new_priorities)
