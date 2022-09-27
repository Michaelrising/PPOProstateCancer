import torch
import numpy as np

################################## set device ##################################

def set_device(gpu=None, cuda=None):
    print("============================================================================================")

    # set device to cpu or cuda
    device = torch.device('cpu')

    if torch.cuda.is_available() and gpu=='cuda' and cuda is not None:
        device = torch.device('cuda:' + str(cuda))
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    elif torch.cuda.is_available() and gpu=='mps':
        device = torch.device('mps')
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")

    print("============================================================================================")
    return device


def evaluate(test_env,init_date, model, eval_times):
    device = model.device
    state_list_all = []
    dose_list_all = []
    rewards_all = []
    actions_all = []
    for _ in range(eval_times):
        state_list = []
        dose_list = []
        fea, state = test_env.reset(init_date)
        state_list.append(state)
        rewards = 0
        actions = []
        while True:
            fea_tensor = torch.from_numpy(np.copy(fea)).to(device).float()
            with torch.no_grad():
                action, _ = model.act(fea_tensor)  # act_exploit(fea_tensor)
            fea, state, reward, done, infos = test_env.step(action.item())
            state_list.append(state)
            rewards += reward
            actions.append(action.item())
            dose_list.append(infos["dose"])
            if done:
                break
        state_list_all.append(state_list)
        dose_list_all.append(dose_list)
        rewards_all.append(rewards)
        actions_all.append(actions)

    mean_rewards = np.mean(rewards_all)
    survival_month = [len(ele) for ele in actions_all]
    pilot_index = survival_month.index(max(survival_month))
    # pilot_index = rewards_all.index(max(rewards_all))
    pilot_actions = actions_all[pilot_index]
    pilot_states_list = state_list_all[pilot_index]
    pilot_dose_list = dose_list_all[pilot_index]
    pilot_states = np.vstack(pilot_states_list)
    pilot_doses = np.vstack(pilot_dose_list).sum(axis=1).astype(bool)
    colors_list = ["red" if tf else "blue" for tf in pilot_doses]
    colors_list.insert(0, "black")
    return mean_rewards, np.mean(survival_month), pilot_actions, pilot_states, colors_list


def greedy_evaluate(test_env, init_date, model, eval_times):
    device = model.device
    state_list_all = []
    dose_list_all = []
    rewards_all = []
    reward_list = []
    actions_all = []
    for _ in range(eval_times):
        state_list = []
        dose_list = []
        fea, state = test_env.reset(init_date)
        state_list.append(state)
        rewards = 0
        actions = []
        while True:
            fea_tensor = torch.from_numpy(np.copy(fea)).to(device).float()
            with torch.no_grad():
                action, _ = model.act_exploit(fea_tensor)
            fea, state, reward, done, infos = test_env.step(action.item())
            state_list.append(state)
            rewards += reward
            reward_list.append(reward)
            actions.append(action.item())
            dose_list.append(infos["dose"])
            if done:
                break
        state_list_all.append(state_list)
        dose_list_all.append(dose_list)
        rewards_all.append(rewards)
        actions_all.append(actions)

    mean_rewards = np.mean(rewards_all)
    survival_month = [len(ele) for ele in actions_all]
    pilot_index = survival_month.index(max(survival_month))
    # pilot_index = rewards_all.index(max(rewards_all))
    pilot_actions = actions_all[pilot_index]
    pilot_states_list = state_list_all[pilot_index]
    pilot_dose_list = dose_list_all[pilot_index]
    pilot_states = np.vstack(pilot_states_list)
    pilot_doses = np.vstack(pilot_dose_list).sum(axis=1).astype(bool)
    colors_list = ["red" if tf else "blue" for tf in pilot_doses]
    colors_list.insert(0, "black")
    return mean_rewards, reward_list, np.mean(survival_month), pilot_actions, pilot_states, colors_list




class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, pars):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, pars)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, pars)
            self.counter = 0

    def save_checkpoint(self, val_loss, pars):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(pars, self.path)
        self.val_loss_min = val_loss