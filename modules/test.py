import torch
import torch.nn as nn
class ActorMod(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorMod, self).__init__()
        self.l1      = nn.Linear(state_dim, n_latent_var)
        self.l1_tanh = nn.Tanh()
        self.l2      = nn.Linear(n_latent_var, n_latent_var)
        self.l2_tanh = nn.Tanh()
        self.l3      = nn.Linear(n_latent_var+60, action_dim)

    def forward(self, input):
        x = self.l1(input)
        x = self.l1_tanh(x)
        x = self.l2(x)
        #return x.softmax(dim=-1)
        out1 = self.l2_tanh(x) # 64x1
        if len(input.shape)==1:
            out2 = input[60:120]   # 60x1 this are the cards on the hand of the player!
            output =torch.cat( [out1, out2], 0)
        else:
            out2 = input[:, 60:120]
            output =torch.cat( [out1, out2], 1) #how to do that?
        print("Input", input.shape, "Out1:", out1.shape, "Out2:", out2.shape)
        x = self.l3(output)
        return x.softmax(dim=-1)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )
        self.equivalent   = ActorMod(state_dim, action_dim, n_latent_var)

works = ActorCritic(180, 60, 64)
action_probs = works.equivalent(torch.rand(180)) # works
#action_probs = works.action_layer(torch.rand(180)) # works
#
action_probs = works.action_layer(torch.rand(20,180)) # works
action_probs = works.equivalent(torch.rand(20, 180)) # works as well
