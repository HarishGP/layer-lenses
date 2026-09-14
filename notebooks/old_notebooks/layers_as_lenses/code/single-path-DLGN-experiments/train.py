import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_dlgn (DLGN_obj, train_data_curr, val_data_curr, test_data_curr,
                train_labels_curr, test_labels_curr, val_labels_curr, config, 
                epochs = 1, plot = False, val_freeze = False, parameter_mask = None, device = 'cpu'
                ):
    
    DLGN_obj.to(device)

    criterion = nn.CrossEntropyLoss(reduction="mean")

    optimizer = optim.SGD(DLGN_obj.parameters(), lr=config["learning_rate"])
    # optimizer = optim.Adam(DLGN_obj.parameters(), lr =lr/50)

    train_data_torch = torch.Tensor(train_data_curr)
    vali_data_torch = torch.Tensor(val_data_curr)
    test_data_torch = torch.Tensor(test_data_curr)

    train_labels_torch = torch.tensor(train_labels_curr, dtype=torch.int64)
    test_labels_torch = torch.tensor(test_labels_curr, dtype=torch.int64)
    vali_labels_torch = torch.tensor(val_labels_curr, dtype=torch.int64)

    num_batches = config["num_batches"]
    batch_size = len(train_data_curr)//num_batches
    losses=[]
    DLGN_obj_store = []
    best_vali_error = len(val_labels_curr)
    DLGN_obj_store.append(deepcopy(DLGN_obj))

    debug_models= []
    train_losses = []
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        for batch_start in range(0,len(train_data_curr),batch_size):
            if (batch_start+batch_size)>len(train_data_curr):
                break			
            inputs = train_data_torch[batch_start:batch_start+batch_size]
            targets = train_labels_torch[batch_start:batch_start+batch_size].reshape(batch_size)
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds = DLGN_obj(inputs).reshape(-1,1)
            outputs = torch.cat((-1*preds, preds), dim=1)
            loss = criterion(outputs, targets)
            train_losses.append(loss.data.cpu().numpy())			
            loss.backward()
        if parameter_mask is not None:
            for name,parameter in DLGN_obj.named_parameters():
                    parameter.grad *= parameter_mask[name].to(device)        
        
        optimizer.step()
        DLGN_obj_store.append(deepcopy(DLGN_obj))
    if plot:
        plt.plot(train_losses)
        print("Train loss is: ",train_losses[-1])

    inputs = test_data_torch
    targets = test_labels_torch
    inputs = inputs.to(device)
    targets = targets.to(device)
    preds = DLGN_obj(inputs).reshape(-1,1)
    outputs = torch.cat((-1*preds, preds), dim=1)
    loss = criterion(outputs, targets)
    print("Test loss is: ",loss)
    return DLGN_obj_store, train_losses

def train_dlgn_2 (DLGN_obj, config, run_specs, train_data,vali_data,test_data,
                train_labels,test_labels,vali_labels,plot=False, parameter_mask = None,
                use_wandb=False):
    # DLGN_obj is the initial network
    # parameter_mask is a dictionary compatible with dict(DLGN_obj.named_parameters())
    # if a key corresponding to a named_parameter is not present it is assumed to be all ones (i.e it will be updated)
    
    device = config['device']
    DLGN_obj.to(device)

    param_groups = [
    {"params": list(DLGN_obj.gating_layers.parameters()), "weight_decay": config['gdecay']}
    # {"params": list(DLGN_obj.value_layers.parameters()), "weight_decay": config['vdecay']}
    ]

    criterion = nn.CrossEntropyLoss(reduction="mean")
    if config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(param_groups, lr=config['lr'], momentum=0.0, nesterov=False)
    elif config['optimizer'] == 'SGDM':
        optimizer = torch.optim.SGD(param_groups, lr=config['lr'], momentum=0.9, nesterov=True)
    elif config['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(param_groups, lr=config['lr'])
    # elif config['optimizer'] == 'Muon':
    #     hidden_weights = [p for p in DLGN_obj.body.parameters() if p.ndim >= 2]
    #     hidden_gains_biases = [p for p in DLGN_obj.body.parameters() if p.ndim < 2]
    #     nonhidden_params = [*DLGN_obj.head.parameters(), *DLGN_obj.embed.parameters()]
    #     param_groups = [
    #         dict(params=hidden_weights, use_muon=True,
    #             lr=0.02, weight_decay=0.01),
    #         dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
    #             lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
    #     ]
    #     optimizer = MuonWithAuxAdam(param_groups)
    else:
        optimizer = None

    scheduler = StepLR(
    optimizer,
    step_size=config['step'],   # epoch at which LR decays
    gamma=0.1
    )

    num_batches = config['num_batches']
    batch_size = len(train_data)//num_batches

    DLGN_obj_store = []
    best_vali_error = len(vali_labels)
    DLGN_obj_store.append(deepcopy(DLGN_obj))

    debug_models= []
    train_losses = []
    found = True
    mid_epoch = -1
    for epoch in tqdm(range(1, config['epochs']+1)):
        DLGN_obj.train()
        optimizer.zero_grad()
        for batch_start in range(0,len(train_data),batch_size):
            if (batch_start+batch_size)>len(train_data):
                break			
            inputs = train_data[batch_start:batch_start+batch_size]
            targets = train_labels[batch_start:batch_start+batch_size].reshape(batch_size)
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds = DLGN_obj(inputs).reshape(-1,1)
            outputs = torch.cat((-1*preds, preds), dim=1)
            loss = criterion(outputs, targets)
            train_losses.append(loss.data.cpu().numpy())			
            loss.backward()
            optimizer.step()
        scheduler.step()
        DLGN_obj_store.append(deepcopy(DLGN_obj))
        if run_specs['print'] and epoch%run_specs['print']==0:
            print(f"Train loss at epoch {epoch}: ",train_losses[-1])
        if found and train_losses[-1] <= 0.67:
            found = False
            mid_epoch = epoch

    if train_losses[-1] < 0.65:
        converged = 'Y'
    else:
        converged = 'N'

    # if plot:
    #     plt.plot(train_losses)
    #     # print("Train loss is: ",train_losses[-1])
    #     # if use_wandb:
    #     #     wandb.log({"train_loss_plot": wandb.Image(plt)})
    #     plt.show()
    #     plt.close()

    # log_dict = {"Train_Loss": train_losses[-1], "Test_Loss": loss, "Converge": conv}
    # if use_wandb:
    #     wandb.log(log_dict)

    return DLGN_obj_store, mid_epoch, converged