import torch
from torch import nn
import numpy as np

def sigmoid(u):
    u = np.maximum(u,-100)
    u = np.minimum(u,100)
    return 1/(1+np.exp(-u))

def ramp_sigmoid(x: torch.Tensor, k: float = 2.0) -> torch.Tensor:
    y = 0.5 + x / (2.0 * k)
    return torch.clamp(y, 0.0, 1.0)

class DLGN_SP(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, hidden_dims=[], beta=100, mode='pwc', device='cpu'):		
        super(DLGN_SP, self).__init__()
        self.num_hidden_layers = len(hidden_dims)
        self.beta=beta  # Soft gating parameter
        self.mode = mode
        self.num_nodes=[input_dim]+hidden_dims+[output_dim]
        self.gating_layers=nn.ModuleList()
        self.value_layers=nn.Parameter(torch.randn([1]+[hidden_dims[0]])/100.)
        self.num_hidden_nodes = hidden_dims
        self.device = device
        for i in range(self.num_hidden_layers+1):
            if i!=self.num_hidden_layers:
                temp = nn.Linear(self.num_nodes[0], self.num_nodes[i+1], bias=False)
                self.gating_layers.append(temp)

    def set_parameters_with_mask(self, to_copy, parameter_masks):
        for (name, copy_param) in to_copy.named_parameters():
            copy_param = copy_param.clone().detach()
            orig_param  = self.state_dict()[name]
            if name in parameter_masks:
                param_mask = parameter_masks[name]>0
                orig_param[param_mask] = copy_param[param_mask]
            else:
                orig_param = copy_param.data.detach()     

    def return_gating_functions(self):
        effective_weights = []
        for i in range(self.num_hidden_layers):
            curr_weight = self.gating_layers[i].weight.detach().clone()
            effective_weights.append(curr_weight)
        return effective_weights
        # effective_weights (and effective biases) is a list of size num_hidden_layers
                            

    def forward(self, x):
        if self.mode=='pwc':
            values=[torch.ones(x.shape).to(self.device)]
        else:
            values=[x]
        
        for i in range(self.num_hidden_layers):
            fiber = [len(x)]+[1]*self.num_hidden_layers
            fiber[i+1] = self.num_hidden_nodes[i]
            fiber = tuple(fiber)
            gate_score = ramp_sigmoid( self.beta*
                  (x@self.gating_layers[i].weight.T))#/torch.norm(self.gating_layers[i].weight, dim=1, keepdim=True).T) # batch * m
            
            if i==0:
                cp = gate_score
            else:
                cp = cp*gate_score #batch * m
            
        return torch.sum(cp*self.value_layers, dim=1)