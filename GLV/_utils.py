import numpy as np
import torch
import matplotlib.pylib as plt

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

def MSEloss_weight(inputs, targets, weight):
    return torch.sum(weight * (inputs - targets) ** 2)


def clip_grad(grad, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:

    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grad) == 0:
        return torch.tensor(0.)
    device = grad.device
    total_norm = torch.norm(grad.detach(), norm_type).to(device)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        grad.detach().mul_(clip_coef.to(grad.device))
    return total_norm

def plot_evolution(Days, PSA, prediction, patientNo, patient_t):
    x = np.arange(prediction.shape[0])
    p, ad, ai = prediction[:, 0], prediction[:, 1], prediction[:, 2]
    plt.scatter(Days, PSA, color="black", marker="*", alpha=0.6)
    plt.plot(x, p, color="black", linestyle="-", linewidth=1)
    plt.xlabel("Time (Days)")
    plt.ylabel("PSA level (ug/ml)")
    plt.savefig("./analysis-dual-sigmoid/model_plots/" + patientNo + "/PSA_" + str(patient_t) + "-" + patientNo + ".png",
                dpi=100)
    # plt.show()
    plt.close()
    plt.plot(x, ad, color="black", linestyle="--", linewidth=1, label="AD")
    plt.plot(x, ai, color="black", linestyle="-.", linewidth=1, label="AI")
    plt.xlabel("Time (Days)")
    plt.ylabel("Cell counts")
    plt.legend(loc='upper right')
    plt.savefig(
        "./analysis-dual-sigmoid/model_plots/" + patientNo + "/Cell_All_" + str(patient_t) + "-" + patientNo + ".png",
        dpi=100)
    # plt.show()
    plt.close()
    plt.plot(x, ai, color="black", linestyle="-.", linewidth=1, label="AI")
    plt.xlabel("Time (Days)")
    plt.ylabel("Cell counts")
    plt.savefig(
        "./analysis-dual-sigmoid/model_plots/" + patientNo + "/Cell_AI_" + str(patient_t) + "-" + patientNo + ".png",
        dpi=100)
    # plt.show()
    plt.close()
