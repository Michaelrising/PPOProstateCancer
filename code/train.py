import os
import glob
import time
from datetime import datetime
import pandas as pd
import torch
import numpy as np
import argparse
import gym
from env.gym_cancer.envs.cancercontrol import CancerControl
from torch.utils.tensorboard import SummaryWriter
# import roboschool
import matplotlib.pyplot as plt
# import pybullet_envs

from PPO import PPO


################################## set device ##################################

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


def evaluate(test_env, model, eval_times):
    device = model.device
    state_list_all = []
    dose_list_all = []
    rewards_all = []
    actions_all = []
    for _ in range(eval_times):
        state_list = []
        dose_list = []
        fea, state = test_env.reset()
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


def greedy_evaluate(test_env, model, eval_times):
    device = model.device
    state_list_all = []
    dose_list_all = []
    rewards_all = []
    actions_all = []
    for _ in range(eval_times):
        state_list = []
        dose_list = []
        fea, state = test_env.reset()
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


################################### Training ###################################

def train(args):
    ################## set device ##################
    device = set_device() if args.cuda_cpu == "cpu" else set_device(args.cuda)

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
    explore_eps = 0.8

    ####################################################
    ################ PPO hyperparameters ################

    decay_step_size = 1000
    decay_ratio = 0.8
    grad_clamp = 0.2
    update_timestep = 2  # update policy every n epoches
    K_epochs = 4  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.00003  # learning rate for actor network
    lr_critic = 0.00001  # learning rate for critic network

    random_seed = args.seed  # set random seed if required (0 = no random seed)

    ########################### Env Parameters ##########################

    if len(str(args.number)) == 1:
        patientNo = "patient00" + str(args.number)
    elif len(str(args.number)) == 2:
        patientNo = "patient0" + str(args.number)
    else:
        patientNo = "patient" + str(args.number)

    K, init_state, pars = args.patients_pars
    weight = np.array([1, 3]) / 4
    base = 1.125
    m1 = args.m1
    m2_ad = args.m2_ad
    m2_ai = args.m2_ai
    drug_decay = args.drug_decay
    drug_length = 8
    ai_end_c = args.ai_end_c
    ad_end_c = args.ad_end_c
    sl = args.sl

    patient = (K, pars, init_state, weight, base, m1, m2_ad, m2_ai, drug_decay, drug_length, ad_end_c, ai_end_c, sl)

    # Create environments.
    envs = [CancerControl(patient=patient) for _ in range(args.num_env)]
    test_env = CancerControl(patient=patient)  # gym.make(args.env_id, patient=patient).unwrapped

    # state space dimension
    state_dim = envs[0].observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = envs[0].action_space.shape[0]
    else:
        action_dim = envs[0].action_space.n

    print("training environment name : " + env_name + "--" + patientNo)
    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten

    log_dir = "../PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    t = datetime.now().strftime("%Y%m%d-%H%M")
    summary_dir = log_dir + '/' + str(patientNo) + '/' + str(t) + "-num_env-" + str(num_env) + "-seed-" + str(
        random_seed) + "-m1-" + str(m1)
    writer = SummaryWriter(log_dir=summary_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir + patientNo))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    # act_log_f_name = log_dir + '/' + str(patientNo) + '/PPO_ActionList_' + env_name + "_" + str(
    #     patientNo) + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    # print("Actions are logged at : " + act_log_f_name)

    #####################################################

    ################### checkpointing ###################

    run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

    directory = "./PPO_pretrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/' + str(patientNo) + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory
    if not os.path.exists(checkpoint_path + "final"):
        os.makedirs(checkpoint_path + "final")
    if not os.path.exists(checkpoint_path + "best"):
        os.makedirs(checkpoint_path + "best")
    checkpoint_format = patientNo + "_" + str(t) + "-m1-" + str(m1) + '-AI-' + str(
        np.around(args.ai_end_c, 1)) + "_PPO_{}_{}_{}.pth".format(env_name, random_seed,
                                                                  run_num_pretrained)  # "-m1-" + str(m1) + "-m2-" + str(m2)
    print("save checkpoint path : " + checkpoint_path)
    print("save checkpoint format : " + checkpoint_format)

    #####################################################

    ############# print all hyperparameters #############

    print("--------------------------------------------------------------------------------------------")

    print("num of envs : " + str(num_env))
    print("max training updating times : ", max_updates)
    print("max timesteps per episode : ", max_ep_len)

    print("model saving frequency : " + str(save_model_freq) + " episodes")
    print("log frequency : " + str(log_freq) + " episodes")
    print("printing average reward over episodes in last : " + str(print_freq) + " episodes")

    print("--------------------------------------------------------------------------------------------")

    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)

    print("--------------------------------------------------------------------------------------------")

    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " episodes")

    else:
        print("Initializing a discrete action space policy")

    print("--------------------------------------------------------------------------------------------")
    print("The initial explore rate : " + str(explore_eps) + " and initial exploit rate is : 1- " + str(explore_eps))

    print("PPO update frequency : " + str(update_timestep) + " episodes")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)

    print("--------------------------------------------------------------------------------------------")
    if args.decayflag:
        print("decaying optimizer with step size : ", decay_step_size, " decay ratio : ", decay_ratio)
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)

    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        for i in range(num_env):
            envs[i].seed(random_seed)
        np.random.seed(random_seed)
        test_env.seed(random_seed)

    #####################################################

    print("============================================================================================")

    ################# training procedure ################

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

    # ppo_agent.load(
    #     "PPO_pretrained/analysis/resistance/patient011_20220330-1135-m1-0.8-AI-0.8_PPO_gym_cancer:CancerControl-v0_0_0.pth")
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    # log_f = open(act_log_f_name, "w+")
    # log_f.write('Sample Actions List, Greedy Actions List\n')

    # ppo_agent.buffers = [ppo_agent.buffer for _ in range(args.num_env)]
    ep_rewards = [0 for _ in range(num_env)]
    # training loop
    reward_record = -1000000
    survival_record = 0
    for i_update in range(max_updates):
        survival_month = 0
        for i, env in enumerate(envs):
            eps = max(- max(i_update - 4000, 0) * (explore_eps - 0.5) / 54000 + explore_eps, 0.5)
            determine = np.random.choice(2, p=[1 - eps, eps])  # explore epsilon
            fea, _ = env.reset()
            ep_rewards[i] = 0
            while True:
                # select action with policy, with torch.no_grad()
                state_tensor, action, action_logprob = ppo_agent.select_action(fea) \
                    if determine else ppo_agent.greedy_select_action(fea)  # state_tensor is the tensor of current state
                ppo_agent.buffers[i].states.append(state_tensor)
                ppo_agent.buffers[i].actions.append(action)
                ppo_agent.buffers[i].logprobs.append(action_logprob)

                fea, _, reward, done, _ = env.step(action)

                # saving reward and is_terminals
                ppo_agent.buffers[i].rewards.append(reward)
                ppo_agent.buffers[i].is_terminals.append(done)

                ep_rewards[i] += reward
                survival_month += 1
                # break; if the episode is over
                if done:
                    break

        mean_rewards_all_env = sum(ep_rewards) / num_env
        mean_survival_month = survival_month / num_env
        # update PPO agent
        if i_update % update_timestep == 0:
            loss = ppo_agent.update(decayflag=args.decayflag, grad_clamp=grad_clamp)

            # log in logging file
            # if i_update % log_freq == 0:
            writer.add_scalar('VLoss', loss, i_update)
            writer.add_scalar("Reward/train", mean_rewards_all_env, i_update)

        # printing average reward
        if i_update % print_freq == 0:
            # print average reward till last episode
            print_avg_reward = mean_rewards_all_env
            print_avg_reward = round(print_avg_reward, 2)
            print_avg_survival_month = round(mean_survival_month, 2)

            print("Updates : {} \t\t SurvivalMonths : {} \t\t Average Reward : {}".format(i_update,
                                                                                          print_avg_survival_month,
                                                                                          print_avg_reward))

        if i_update % eval_interval == 0:
            # rewards, survivalMonth, actions, states, colors = evaluate(test_env, ppo_agent.policy_old, eval_times)
            g_rewards, g_survivalMonth, g_actions, g_states, g_colors = greedy_evaluate(test_env, ppo_agent.policy_old,
                                                                                        eval_times)
            # writer.add_scalar("Reward/evaluate", rewards, i_update)
            writer.add_scalar("Reward/greedy_evaluate", g_rewards, i_update)
            writer.add_scalar("Survival month", g_survivalMonth, i_update)
            # log_f.write('{},{}\n'.format(actions, g_actions))
            # log_f.flush()
            if g_rewards > reward_record and i_update >= model_save_start_updating_steps:
                reward_record = g_rewards
                x = np.arange(g_states.shape[0])
                ad = g_states[:, 0]
                ai = g_states[:, 1]
                psa = g_states[:, 2]
                fig = plt.figure(figsize=(15, 5))
                ax1 = fig.add_subplot(1, 3, 1)
                ax1.plot(x, psa, color="black", linestyle="-", linewidth=1)
                plt.scatter(x, psa, color=g_colors)
                ax1.set_xlabel("Time (months)")
                ax1.set_ylabel("PSA level (ug/ml)")

                ax2 = fig.add_subplot(1, 3, 2)
                ax2.plot(x, ad, color="black", linestyle="--", linewidth=1, label="AD")
                ax2.plot(x, ai, color="black", linestyle="-.", linewidth=1, label="AI")
                ax2.set_xlabel("Time (months)")
                ax2.set_ylabel("Cell counts")
                ax2.legend(loc='upper right')

                phi, gamma = pars[-3:-1]
                a = 1 / (1 + np.exp(-gamma * x / 12.)).reshape(-1)
                c1 = (ai * a / K[0]) ** phi
                c2 = (ad * a / K[1]) ** phi
                ax3 = fig.add_subplot(1, 3, 3)
                ax3.plot(x, c1, color="black", linestyle="--", linewidth=1, label="c1:AI to AD")
                ax3.plot(x, c2, color="black", linestyle="-.", linewidth=1, label="c2:AD to AI")
                ax2.set_xlabel("Time (months)")
                ax2.set_ylabel("Competition intensity")
                ax2.legend(loc='upper right')
                plt.savefig(summary_dir + "/best_model.png", dpi=150)
                plt.close()

                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path + "best/" + checkpoint_format)
                ppo_agent.save(checkpoint_path + "best/" + checkpoint_format)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # save model weights
            if i_update % save_model_freq == 0:
                x = np.arange(g_states.shape[0])
                ad = g_states[:, 0]
                ai = g_states[:, 1]
                psa = g_states[:, 2]
                fig = plt.figure(figsize=(15, 5))
                ax1 = fig.add_subplot(1, 3, 1)
                ax1.plot(x, psa, color="black", linestyle="-", linewidth=1)
                plt.scatter(x, psa, color=g_colors)
                ax1.set_xlabel("Time (months)")
                ax1.set_ylabel("PSA level (ug/ml)")

                ax2 = fig.add_subplot(1, 3, 2)
                ax2.plot(x, ad, color="black", linestyle="--", linewidth=1, label="AD")
                ax2.plot(x, ai, color="black", linestyle="-.", linewidth=1, label="AI")
                ax2.set_xlabel("Time (months)")
                ax2.set_ylabel("Cell counts")
                ax2.legend(loc='upper right')

                phi, gamma = pars[-3:-1]
                a = 1 / (1 + np.exp(-gamma * x/12)).reshape(-1)
                c1 = (ai * a / K[0]) ** phi
                c2 = (ad * a / K[1]) ** phi
                ax3 = fig.add_subplot(1, 3, 3)
                ax3.plot(x, c1, color="black", linestyle="--", linewidth=1, label="c1")
                ax3.plot(x, c2, color="black", linestyle="-.", linewidth=1, label="c2")
                ax3.set_xlabel("Time (months)")
                ax3.set_ylabel("Competition intensity")
                ax3.legend(loc='upper right')
                plt.savefig(summary_dir + "/final_model.png", dpi=150)
                plt.close()

                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path + "final/" + checkpoint_format)
                ppo_agent.save(checkpoint_path + "final/" + checkpoint_format)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

    # log_f.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


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
    parser.add_argument('--type', type=float, default=1)
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
    ending_states = ending_states_all_patients.loc[args.number]
    args.m2_ad, args.m2_ai, args.ad_end_c, args.sl = ending_states.m_ad, ending_states.m_ai, ending_states.c_ad, ending_states.sl
    if args.number == 32:
        args.m2_ad = 1.
    if args.number in [11, 12, 19, 25, 36, 54, 85, 88, 99, 101]: # 52 type = 1
        args.type = 2.
    if len(str(args.number)) == 1:
        patientNo = "patient00" + str(args.number)
    elif len(str(args.number)) == 2:
        patientNo = "patient0" + str(args.number)
    else:
        patientNo = "patient" + str(args.number)
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
        if arg.endswith('.csv'):
            pars_df = pd.read_csv(parsdir + patientNo + '/' + arg)
            _, K, states, pars, best_pars = [np.array(pars_df.loc[i, ~np.isnan(pars_df.loc[i, :])]) for i in range(5)]
            PARS_LIST.append(best_pars)
    # Init = np.array([mean_v / Mean_psa * true_psa[0] / cell_size, 1e-4 * K[1], true_psa[0]])
    Init_states = states[:3]
    PARS_ARR = np.stack(PARS_LIST)
    pars = np.mean(PARS_ARR, axis=0)
    args.patients_pars = (K, Init_states, pars)
    train(args)








