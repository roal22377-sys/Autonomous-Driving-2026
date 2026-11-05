from torch import nn
from torch import distributions as pyd
import torch.nn.functional as F
import torch
import math

LOG_STD_MAX = 2
LOG_STD_MIN = -4
EPS = 1e-8

def initWeights(m, init_bias=0.0):
    if isinstance(m, torch.nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
        m.bias.data.normal_(init_bias, 0.01)

########################## Distribution ###########################
class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self._loc = loc
        self._scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def loc(self):
        mu = self._loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu
    
    def entropy(self):
        sample = self.rsample()
        log_prob = self.log_prob(sample).sum(-1, keepdim=True)
        return -log_prob.mean() 


############################## Actor ##############################

class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()

        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.hidden1_units = args.hidden_dim
        self.hidden2_units = args.hidden_dim
        self.log_std_init = args.log_std_init
        self.activation = args.activation

        # ======= implement here ======= #
        self.fc1 = nn.Linear(self.obs_dim, self.hidden1_units)
        self.fc2 = nn.Linear(self.hidden1_units, self.hidden2_units)

        self.fc_mean = nn.Linear(self.hidden2_units, self.action_dim)
        # ============================== #
        self.activ = eval(f'torch.nn.{self.activation}()')
        self.output_activ = torch.tanh
        self.log_std = torch.tensor(
            [self.log_std_init]*self.action_dim, dtype=torch.float32, 
            requires_grad=args.log_std_grad, device=args.device
        )
        self.log_std = nn.Parameter(self.log_std)
        self.register_parameter(name="my_log_std", param=self.log_std)

    def forward(self, x):
        # ======= implement here ======= #
        x = self.activ(self.fc1(x))
        x = self.activ(self.fc2(x))
        # ============================== #
        mean = self.output_activ(self.fc_mean(x))

        log_std = torch.clamp(self.log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        std = torch.ones_like(mean)*torch.exp(log_std)
        
        action_dists = torch.distributions.Normal(mean, std)
        return action_dists

    def initialize(self):
        self.apply(initWeights)
        

class SquashedActor(nn.Module):
    def __init__(self, args):
        super(SquashedActor, self).__init__()
        
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.hidden1_units = args.hidden_dim
        self.hidden2_units = args.hidden_dim
        self.activation = args.activation

        # ======= implement here ======= #
        self.fc1 = nn.Linear(self.obs_dim, self.hidden1_units)
        self.fc2 = nn.Linear(self.hidden1_units, self.hidden2_units)
        
        self.mean_layer = nn.Linear(self.hidden2_units, self.action_dim)
        self.log_std_layer = nn.Linear(self.hidden2_units, self.action_dim)
        # ============================== #
        self.activ = eval(f'torch.nn.{self.activation}()')
        
    def forward(self, x):
        # ======= implement here ======= #
        x = self.activ(self.fc1(x))
        x = self.activ(self.fc2(x))
        # ============================== #
        mean = self.mean_layer(x)
        
        log_std = self.log_std_layer(x)
        std = torch.exp(torch.clamp(log_std, min=LOG_STD_MIN*2.5, max=LOG_STD_MAX))
        
        action_dists = SquashedNormal(mean, std)
        return action_dists

    def initialize(self):
        self.apply(initWeights)
 

############################## Critic ##############################

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()

        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.hidden1_units = args.hidden_dim
        self.hidden2_units = args.hidden_dim
        self.activation = args.activation

        # ======= implement here ======= #
        self.fc1 = nn.Linear(self.obs_dim, self.hidden1_units)
        self.fc2 = nn.Linear(self.hidden1_units, self.hidden2_units)
        self.fc3 = nn.Linear(self.hidden2_units, 1)
        # ============================== #
        self.activ = eval(f'torch.nn.{self.activation}()')

    def forward(self, x):
        # ======= implement here ======= #
        x = self.activ(self.fc1(x))
        x = self.activ(self.fc2(x))
        x = self.fc3(x)
        # ============================== #
        assert x.dim() == 2
        return x

    def initialize(self):
        self.apply(initWeights)
        
class QCritic(nn.Module):
    def __init__(self, args):
        super(QCritic, self).__init__()

        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.hidden1_units = args.hidden_dim
        self.hidden2_units = args.hidden_dim
        self.activation = args.activation

        # ======= implement here ======= #
        self.fc1 = nn.Linear(self.obs_dim + self.action_dim, self.hidden1_units)
        self.fc2 = nn.Linear(self.hidden1_units, self.hidden2_units)
        self.fc3 = nn.Linear(self.hidden2_units, 1)
        # ============================== #
        self.activ = eval(f'torch.nn.{self.activation}()')

    def forward(self, state, action):
        # ======= implement here ======= #
        x = torch.cat([state, action], dim=-1)
        x = self.activ(self.fc1(x))
        x = self.activ(self.fc2(x))
        x = self.fc3(x)
        # ============================== #
        assert x.dim() == 2
        return x

    def initialize(self):
        self.apply(initWeights)
        
class DoubleQCritic(nn.Module):
    def __init__(self, args):
        super(DoubleQCritic, self).__init__()
        
        self.critic1 = QCritic(args)
        self.critic2 = QCritic(args)
        
    def forward(self, state, action):
        return self.critic1(state, action), self.critic2(state, action)
    
    def minimum(self, state, action):
        return torch.minimum(self.critic1(state, action), self.critic2(state, action))

    def initialize(self):
        self.critic1.initialize()
        self.critic2.initialize()