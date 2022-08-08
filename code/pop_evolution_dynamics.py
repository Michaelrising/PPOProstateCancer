import os
import glob
import time
from datetime import datetime
import pandas as pd
import torch
import numpy as np
import argparse
import gym
import matplotlib.pyplot as plt
# import pybullet_envs
import seaborn as sns
from matplotlib.lines import Line2D
import sys
sys.path.append(".")
from env.gym_cancer.envs.cancercontrol import CancerControl
from PPO import PPO
def set_device(cuda=None):
    print("============================================================================================")

    # set device to cpu or cuda
    device = torch.device('cpu')

    if torch.cuda.is_available() and cuda is not None:
        device = torch.device('cuda:' + str(cuda))
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")

    print("============================================================================================")
    return device


#################################### Testing ###################################


def plot_figure(data, save_path,patientNo, par = 0):
    if par:
        save_name = "_best_survival_time.eps"
    else:
        save_name = '_best_reward.eps'
    states = data["states"]
    doses = data["doses"]
    cs = sns.color_palette('Paired')
    x = np.arange(states.shape[0]) * 28
    ad = states[:, 0]
    ai = states[:, 1]
    psa = states[:, 2]
    fig = plt.figure(figsize=(5, 4))
    # plt.style.use('seaborn')
    plt.style.use(['science', "nature"])
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(x, psa, linestyle="-", linewidth=1.2, c=cs[1])
    # ax1.plot([0,850], [1.37, 1.37], ls='--', lw=3, color=cs[4])
    # # plt.text(850, 1.37, str(1.37), fontdict={'size': 25, 'color': 'black', "family": 'Times New Roman'})
    # ax1.plot([900, 2200], [2.37, 2.37], ls='--', lw=3, color=cs[6])
    # # plt.text(2200, 2.37, str(2.37), fontdict={'size': 25, 'color': 'black', "family": 'Times New Roman'})
    # plt.scatter(x[101], psa[101], color=cs[5], marker='*', s=200)
    nadir_psa = min(psa)
    nadir_psa_x = x[np.where(psa == nadir_psa)[0]]
    # plt.scatter(nadir_psa_x, nadir_psa, color=cs[5], marker='*', s = 180)
    # ax1.set_xlabel("Time (Days)",  fontsize=25)
    ax1.set_ylabel("PSA ($\mu$g/L)",  fontsize=15)
    ax1.tick_params(labelsize = 12)
    ax1.set_xticks(ticks = [], minor=False)
    ax1.set_xticklabels(labels=[], minor=False)
    threshold = Line2D([0], [0], color=cs[5], ls='--', lw=1.2)
    ax1.legend([threshold], ['trend of threshold'], fontsize=12, loc=0)

    # ax1.set_yticks(ticks = [5, 7.5, 10])
    # ax1.set_yticklabels(labels = [5, 7.5, 10], minor = False)
    # plt.text(-x[-1] * 0.17, max(psa) * 1.1, "d", fontdict={'size': 32, 'color': 'black', "family": 'Times New Roman'},
    #          weight='bold')

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(x, ad, label="response",  lw=1.2, c=cs[1])
    ax2.plot(x, ai, label="resistance",  lw=1.2, c=cs[3])
    # plt.scatter(nadir_psa_x, ad[np.where(psa == nadir_psa)[0]], color=cs[5], marker='*', s=180)
    # plt.annotate("Transition Point", xy=(nadir_psa_x, ad[np.where(psa == nadir_psa)[0]]),
    #              xytext = (nadir_psa_x * 1.8, 1.5*ad[np.where(psa == nadir_psa)[0]]),
    #              arrowprops=dict(facecolor=cs[5], shrink=0.05, edgecolor = cs[5]),
    #              color= cs[5],fontsize =18)
    ax2.set_xlabel("Time (Days)",  fontsize=15)
    ax2.set_ylabel("$\#$ of Cell",  fontsize=15)
    ax2.tick_params(labelsize=12)
    # ax2.set_yticks(ticks = [0, 2.5 * 10**9, 5 * 10**9])
    # ax2.set_yticklabels(labels=[0, 2.5, 5])
    # plt.text(0.0, 1, '$10^{9} $', fontsize = 8, transform=ax2.transAxes)
    ax2.legend(loc=0, fontsize=12, ncol=1)
    plt.tight_layout()
    plt.savefig(save_path + "_evolution" + save_name, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    COLOR_CPA = cs[5]
    COLOR_LEU = cs[1]
    x_dose = np.arange(0, doses.shape[0]) * 28
    doses[:,0] = doses[:, 0]/200
    doses[:,1] = doses[:,1]/7.5
    dose = doses.sum(axis=1)
    cumsum_dose1 = np.cumsum(doses[:,0])
    cumsum_dose2 = np.cumsum(doses[:,1])
    fig = plt.figure(figsize=(5, 4))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = ax1.twinx()
    ax1.plot(x_dose, cumsum_dose1, color=cs[1], lw=1, zorder=3)
    ax1.plot(x_dose, cumsum_dose2, color=cs[5], lw=1, zorder=3)
    # plt.yticks(ticks=[0, 10, 20, 30], fontsize=15)
    ax1.tick_params(axis="y", labelsize=15)
    ax1.set_xticks([])
    ax1.set_zorder(3)
    ax2.bar(x_dose, height=doses[:, 0], width=28, color=cs[0], zorder=0)
    ax2.bar(x_dose, bottom=doses[:, 0], height=doses[:, 1], width=28, color=cs[4], zorder=0)
    ax2.set_ylim(0, 4)
    ax2.set_zorder(2)
    ax2.tick_params(axis="y", labelsize=15)
    # read clinical dosages into the plot
    clinical_data = pd.read_csv("../data/dataTanaka/Bruchovsky_et_al/" + patientNo + ".txt", header=None)
    cpa = np.array(clinical_data[2])
    cpa[np.isnan(cpa)] = 0
    leu = np.array(clinical_data[3])
    leu[np.isnan(leu)] = 0
    days = np.array(clinical_data[8])
    days = days-days[0]
    days_diff = np.diff(days)
    days_diff = np.append(days_diff, 28)
    cpa_month_normalized = cpa/200
    leu_month_normalized = leu/7.5
    leu_month_normalized[np.where(leu_month_normalized>1)[0]] = 1
    cumsum_dose1_clinical = np.cumsum(cpa_month_normalized*days_diff/28)
    cumsum_dose2_clinical = np.cumsum(leu_month_normalized)
    ax3 = fig.add_subplot(2, 1, 2)
    ax4 = ax3.twinx()
    ax3.plot(days, cumsum_dose1_clinical, c=cs[1], lw=1, zorder=3)
    ax3.plot(days, cumsum_dose2_clinical, c=cs[5], lw=1, zorder=3)
    ax3.set_xlim(ax1.get_xlim())
    plt.xticks(fontsize=12)
    # plt.yticks(ticks=[0, 10, 20, 30], fontsize=15)
    ax3.tick_params(axis="both", labelsize=15)
    ax3.set_xlabel("Time (Days)", fontsize=15)
    ax3.set_zorder(3)
    ax4.bar(days,cpa_month_normalized, width=days_diff, color=cs[0], zorder=0)
    ax4.bar(days, leu_month_normalized, bottom=cpa_month_normalized, width=days_diff, color=cs[4], zorder=0)
    ax4.set_ylim(0, 4)
    ax4.set_zorder(2)
    ax4.tick_params(axis="y", labelsize=15)
    ax4.set_ylabel('', fontsize=15)
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path + 'cumulative__dosages' + save_name, dpi=300,bbox_inches='tight')
    # ax1 = fig.add_subplot(111)
    # ax2 = ax1.twinx()
    # ax1.plot(x_dose, np.cumsum(doses[:,0]), color=COLOR_CPA, label='CPA', lw=2)
    # ax1.set_xlabel("Days",  fontsize=13)
    # ax1.set_ylabel("CPA", fontsize=13)
    # # ax1.set_ylim(-5, 205)
    # ax1.set_yticks(ticks = np.arange(0, 10000, 2500), labels=[0, 2.5, 5., 7.5])
    # ax1.tick_params(axis="y", labelcolor=COLOR_CPA,  labelsize=13)
    # ax1.legend(loc=(0.03, 0.9), fontsize=13, facecolor=COLOR_CPA)
    # plt.grid(False)
    # ax2.plot(x_dose,  np.cumsum(doses[:,1]), color=COLOR_LEU, label='LEU', lw=2, ls='--')
    # ax2.set_ylabel("LEU", fontsize=13)
    # ax2.tick_params(axis="y", labelcolor=COLOR_LEU, labelsize=13)
    # ax2.tick_params(axis="x", labelsize=13)
    # # ax2.set_yticks(ticks = np.arange(0, 22.5, 7.5))
    # # plt.ylim(-.5, 15)
    # ax2.legend(loc=(0.03, 0.8), fontsize=13, facecolor=COLOR_LEU)
    # plt.savefig(save_path+'_dosages' + save_name, dpi=300)
    plt.show()
    plt.close()
    return

def test(args, file, t):
    ####### initialize environment hyperparameters ######

    env_name = args.env_id  # "RoboschoolWalker2d-v1"
    num_env = args.num_env
    max_updates = args.max_updates
    eval_interval = args.eval_interval
    model_save_start_updating_steps = args.model_save_start_updating_steps
    eval_times = args.eval_times

    has_continuous_action_space = False  # continuous action space; else discrete

    max_ep_len = 120  # max timesteps in one episode

    print_freq = 2  # print avg reward in the interval (in num updating steps)
    log_freq = 2  # log avg reward in the interval (in num updating steps)
    save_model_freq = eval_interval * 10  # save model frequency (in num updating steps)

    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num updating steps)

    ####################################################
    ################ PPO hyperparameters ################

    decay_step_size = 500
    decay_ratio = 0.5
    update_timestep = 1  # update policy every n timesteps
    K_epochs = 4  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.00003  # learning rate for actor network
    lr_critic = 0.0001  # learning rate for critic network

    random_seed = args.seed  # set random seed if required (0 = no random seed)

    ########################### Env Parameters ##########################

    if len(str(args.number)) == 1:
        patientNo = "patient00" + str(args.number)
    elif len(str(args.number)) == 2:
        patientNo = "patient0" + str(args.number)
    else:
        patientNo = "patient" + str(args.number)
    # patientNo ="patient006"
    K, init_states, pars = args.patients_pars
    weight = np.ones(2) / 2
    base = 1.125
    m1 = args.m1
    m2_ad = args.m2_ad
    m2_ai = args.m2_ai
    drug_decay = args.drug_decay
    drug_length = 8
    ai_end_c = args.ai_end_c
    ad_end_c = args.ad_end_c
    sl = args.sl

    patient = (K, pars, init_states, weight, base, m1, m2_ad, m2_ai, drug_decay, drug_length, ad_end_c, ai_end_c, sl)

    test_env = CancerControl(patient=patient, mode='test')

    # state space dimension
    state_dim = test_env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = test_env.action_space.shape[0]
    else:
        action_dim = test_env.action_space.n


    # initialize a PPO agent
    ppo_agent = PPO(
            state_dim,
            action_dim,
            lr_actor,
            lr_critic,
            gamma,
            K_epochs,
            eps_clip,
            has_continuous_action_space,
            num_env,
            device,
            decay_step_size,
            decay_ratio,
            action_std)

    checkpoint_path = file
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")
    if not os.path.exists("../PPO_figs/analysis/"):
        os.makedirs("../PPO_figs/analysis/")
    if not os.path.exists("../PPO_policy/analysis/"):
        os.makedirs("../PPO_policy/analysis/")
    if not os.path.exists("../PPO_states/analysis/"):
        os.makedirs("../PPO_states/analysis/")
    record_states_high_reward = 0
    record_dose_high_reward = 0
    record_states_high_survival_time = 0
    record_dose_survival_month = 0
    test_running_reward = 0
    total_test_episodes = 1
    states = []
    doses = []
    evolution=[]
    record_reward = -1000
    record_survival_month = 0
    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        ep_survival_month = 0
        fea, state = test_env.reset()
        states.append(state)
        while True:
            _, action, _ = ppo_agent.greedy_select_action(fea)
            fea, state, reward, done, infos = test_env.step(action)
            states.append(state)
            doses.append(infos["dose"])
            evolution.append(infos['evolution'])
            ep_reward += reward
            ep_survival_month += 1
            if done:
                break
        if ep==1:
            evolution_ = np.concatenate(evolution, axis=1).T
            pd.DataFrame(evolution_, columns=['ad', 'ai', 'psa']).to_csv("../PPO_states/analysis/" + patientNo + "_evolution_states.csv")
            doses_ = np.stack(doses)
            pd.DataFrame(doses_, columns=["CPA", "LEU"]).to_csv(
                "../PPO_policy/analysis/" + patientNo + "_rl_dosages.csv")
        if record_reward < ep_reward:
            record_reward = ep_reward
            record_states_high_reward = np.vstack(states.copy())
            record_dose_high_reward = np.vstack(doses)
        if record_survival_month < ep_survival_month:
            record_survival_month = ep_survival_month
            record_states_high_survival_time = np.vstack(states.copy())
            record_dose_survival_month = np.vstack(doses)
        states.clear()
        doses.clear()
        test_running_reward += ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))

    test_env.close()
    # maximum rewards
    High_reward = {"states": record_states_high_reward, "doses": record_dose_high_reward}
    High_survival = {"states": record_states_high_survival_time, "doses": record_dose_survival_month}

    savepath = "../PPO_figs/analysis/" +patientNo
    plot_figure(High_reward, savepath ,patientNo, 0)
    # plot_figure(High_survival, savepath, 1)
    # # pd.DataFrame(record_states_high_reward).to_csv(
    # #     "./PPO_states/analysis/" + patientNo + "_converge_high_reward_states.csv")
    # # pd.DataFrame(record_states_high_survival_time).to_csv(
    # #     "./PPO_states/analysis/" + patientNo + "_converge_high_survival_states.csv")
    pd.DataFrame(record_dose_high_reward).to_csv("../PPO_policy/analysis/" + patientNo + "_rl_dosages.csv")
    # pd.DataFrame(record_dose_survival_month).to_csv("./PPO_policy/analysis/" + patientNo + "_converge_high_survival_dosages.csv")
    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")
    return avg_test_reward

if __name__ == '__main__':

    print("============================================================================================")

    parsdir = "./GLV/analysis-dual-sigmoid/model_pars/"
    env_dict = gym.envs.registration.registry.env_specs.copy()
    for env in env_dict:
        if 'CancerControl-v0' in env:
            print("Remove {} from registry".format(env))
            del gym.envs.registration.registry.env_specs[env]
    ending_states_all_patients = pd.read_csv('./Analysis/end_states_all_patients.csv', index_col=0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join('config', 'sacd.yaml'))
    parser.add_argument('--env_id', type=str, default='gym_cancer:CancerControl-v0')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument("--cuda_cpu", type=str, default="cuda", help="Set device as cuda or cpu")
    parser.add_argument('--m1', type=float, default=0.8)
    parser.add_argument('--m2_ai', type=int, default=12)
    parser.add_argument('--m2_ad', type=int, default=12)
    parser.add_argument('--drug_decay', type=float, default=0.75, help="The decay rate for drug penalty")
    parser.add_argument('--ai_end_c', type=float, default=0.8,
                        help='The concentration of AI phenotype for determining the simulation end')
    parser.add_argument('--ad_end_c', type=float, default=0.9,
                        help='The concentration of AI phenotype for determining the simulation end')
    parser.add_argument('--sl', type=float, default=120,
                        help='survival month for patients')
    parser.add_argument('--seed', type=int, default=0)  # random.randint(0,100000))
    parser.add_argument('--patients_pars', type=tuple, default=(0,))
    parser.add_argument('--patients_train', type=list, default=[])
    parser.add_argument('--number', '-n', type=int, help='Patient No., int type, requested',
                        default=1)  # the only one argument needed to be inputted
    parser.add_argument('--num_env', type=int, help='number of environments',
                        default=2)
    parser.add_argument('--max_updates', type=int, help='max number of updating times',
                        default=int(1e5))
    parser.add_argument("--eval_interval", type=int, help="interval to evaluate the policy and plot figures",
                        default=50)
    parser.add_argument('--decayflag', type=bool, default=True, help='lr decay flag')
    parser.add_argument('--model_save_start_updating_steps', type=int, default=500,
                        help="The start steps of saving best model")
    parser.add_argument("--eval_times", type=int, default=10, help='The evaluation time of current policy')
    args = parser.parse_args()

    device = set_device() if args.cuda_cpu == "cpu" else set_device(args.cuda)
    type = ['response', 'resistance']
    AVA_REWARD = {}
    for t in type:
        a = os.listdir('../PPO_pretrained/')
        print(a)
        analysis = os.listdir('../PPO_pretrained/analysis/' + t)
        analysis.sort()
        for file in analysis:
            Number = int(file[7:10])
            args.number = Number
            patientNo = file[:10]
            # if len(str(args.number)) == 1:
            #     patientNo = "patient00" + str(args.number)
            # elif len(str(args.number)) == 2:
            #     patientNo = "patient0" + str(args.number)
            # else:
            #     patientNo = "patient" + str(args.number)
            ending_states = ending_states_all_patients.loc[args.number]
            args.m2_ad, args.m2_ai, args.ad_end_c, args.sl = ending_states.m_ad, ending_states.m_ai, ending_states.c_ad, ending_states.sl
            parslist = os.listdir(parsdir + patientNo)
            clinical_data = pd.read_csv("../data/dataTanaka/Bruchovsky_et_al/" + patientNo + ".txt", header=None)
            true_psa = np.array(clinical_data.loc[:, 4])
            true_psa = true_psa[~np.isnan(true_psa)]
            cell_size = 5.236e-10
            mean_v = 5
            Mean_psa = 22.1
            PARS_LIST = []
            # reading the ode parameters and the initial/terminal states
            for arg in parslist:
                pars_df = pd.read_csv(parsdir + patientNo + '/' + arg)
                _, K, states, pars, best_pars = [np.array(pars_df.loc[i, ~np.isnan(pars_df.loc[i, :])]) for i in range(5)]
                PARS_LIST.append(best_pars)
            Init = states[:3]
            PARS_ARR = np.stack(PARS_LIST)
            pars = np.mean(PARS_ARR, axis=0)
            args.patients_pars = (K, Init, pars)

            print("============================================================================================")
            model_loc = '../PPO_pretrained/analysis/' + t +'/' +file
            reward = test(args, model_loc, t)
            AVA_REWARD[patientNo] = reward

    AVA_REWARD = pd.DataFrame.from_dict(AVA_REWARD, orient='index')
    AVA_REWARD.to_csv('./Analysis/HK_average_rewards_all_patients.csv')