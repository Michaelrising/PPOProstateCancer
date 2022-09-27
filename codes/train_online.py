# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import gym
import numpy as np
import torch.cuda

from env.gym_cancer.envs.cancercontrol import CancerControl
from _utils import *
from GLV.glv_train_online import *
from PPO import PPO
import multiprocessing as mp
import pandas as pd
import xitorch
import matplotlib.pyplot as plt



################################### Training ###################################

def train_online(args, clinical_data):
    ################## set device ##################
    device = set_device(args.gpu_cpu, args.cuda)

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

    online_tuple = (patientNo, args.pa_t)
    init_date, K, init_state, real_pars, online_pars = args.patients_pars
    fixed_init_date = init_date
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
    max_episodes_steps = 120 - int(init_date.item()/28)

    predict_term = 3

    real_patient = (K, real_pars, init_state, weight, base, m1, m2_ad, m2_ai, drug_decay, drug_length, ad_end_c, ai_end_c, sl, max_episodes_steps)
    online_patient = (K, online_pars, init_state, weight, base, m1, m2_ad, m2_ai, drug_decay, drug_length, ad_end_c, ai_end_c, sl, max_episodes_steps)
    # Create environments.
    # Real patient use the model trained with all the longitudial data
    # online patient only use the first circle data which view as a new patient
    # and the PPO is applied in the online patient one step and validate in the real patient
    real_envs = CancerControl(patient=real_patient, t=init_date.item())
    online_envs = [CancerControl(patient=online_patient, t=init_date.item()) for _ in range(args.num_env)]
    test_env = CancerControl(patient=real_patient, t=init_date.item())

    # Generate the evolution curve for the clinical practice

    # state space dimension
    state_dim = online_envs[0].observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = online_envs[0].action_space.shape[0]
    else:
        action_dim = online_envs[0].action_space.n

    print("training environment name : " + env_name + "--" + patientNo)
    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten

    log_dir = "../PPO_logs/online"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    t = datetime.now().strftime("%Y%m%d-%H%M")
    summary_dir = log_dir + '/' + str(patientNo) + '/' + str(t)  + "/patient_clone" + str(args.pa_t) +'_'  + "-num_env-" + str(num_env) + "-seed-" + str(
        random_seed) + "-m1-" + str(m1)
    writer = SummaryWriter(log_dir=summary_dir)
    glv_dir = log_dir + '/' + str(patientNo) + '/' + str(t) + '/glv_online'
    mk_dir(patientNo=patientNo, glv_dir=glv_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir + patientNo))[2]
    run_num = len(current_num_files)

    print("current logging run number for " + env_name + " : ", run_num)
    # print("Actions are logged at : " + act_log_f_name)

    #####################################################

    ################### checkpointing ###################

    run_num_pretrained = 0  #### change this to prevent overwriting weights in same env_name folder

    directory = "../PPO_pretrained/online"
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
    checkpoint_format = patientNo + "_" + str(args.pa_t) + '_' + str(t) + "-m1-" + str(m1) + '-AI-' + str(
        np.around(args.ai_end_c, 1)) + "_PPO_{}_{}_{}.pth".format(env_name, random_seed,
                                                                  run_num_pretrained)  # "-m1-" + str(m1) + "-m2-" + str(m2)
    print("save checkpoint path : " + checkpoint_path)
    print("save checkpoint format : " + checkpoint_format)

    #####################################################
    # glv_parser = argparse.ArgumentParser(description='Patient arguments')
    # glv_parser.add_argument('--number', default=args.number, help='Patient No., int type, requested', type=int)
    # glv_parser.add_argument('--g_mode', default=args.g_mode, help='Setting online or offline learning', type=str)
    # glv_parser.add_argument('--t', default=args.pa_t, type=int)
    # glv_parser.add_argument('--patientNo', default=args.patientNo, help='Patient No., str type, requested', type=str)
    # glv_args = glv_parser.parse_args()

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
            online_envs[i].seed(random_seed)
            online_envs[i].seed(random_seed)
        np.random.seed(random_seed)
        test_env.seed(random_seed)

    #####################################################
    num_cores = int(mp.cpu_count())
    print("The local computer has : " + str(num_cores) + " cores")
    pool = mp.Pool(num_env)

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
    rewards_deque = collections.deque(maxlen=200)
    survival_deque = collections.deque(maxlen=200)
    _mean_rewards_all_env = -1000000
    _mean_survival_month = 1
    reward_record = -1000000
    survival_record = 0
    treated_rewards = 0
    treated_months = 0
    treated_actions = []
    treated_states = []
    treated_colors = []
    new_cpa = []
    new_leu = []
    new_psa = []
    new_days = []
    record_flag = False
    online_updates_epoch = 2e3
    for i_update in range(max_updates):
        survival_month = 0
        for i, online_env in enumerate(online_envs):
            eps = max(- max(i_update - 4000, 0) * (explore_eps - 0.5) / max_updates/2 + explore_eps, 0.5)
            determine = np.random.choice(2, p=[1 - eps, eps])  # explore epsilon
            fea, _ = online_env.reset(init_date.item())
            ep_rewards[i] = 0
            while True:
                # select action with policy, with torch.no_grad()
                state_tensor, action, action_logprob = ppo_agent.select_action(fea) \
                    if determine else ppo_agent.greedy_select_action(fea)  # state_tensor is the tensor of current state
                ppo_agent.buffers[i].states.append(state_tensor)
                ppo_agent.buffers[i].actions.append(action)
                ppo_agent.buffers[i].logprobs.append(action_logprob)

                fea, _, reward, done, _ = online_env.step(action)

                # saving reward and is_terminals
                ppo_agent.buffers[i].rewards.append(reward)
                ppo_agent.buffers[i].is_terminals.append(done)
                ep_rewards[i] += reward
                survival_month += 1
                # break; if the episode is over
                if done:
                    break

        mean_rewards_all_env = sum(ep_rewards) / num_env + treated_rewards/num_env
        mean_survival_month = survival_month / num_env + treated_months/num_env
        if survival_record < mean_survival_month:
            survival_record = mean_survival_month
        if mean_rewards_all_env > 0 and record_flag == False:
            record_flag = True
        if record_flag == True:
            rewards_deque.append((mean_rewards_all_env - _mean_rewards_all_env) / _mean_rewards_all_env)
            survival_deque.append(mean_survival_month)


        # printing average reward

        if i_update % print_freq == 0:
            # print average reward till last episode
            print_avg_reward = round(mean_rewards_all_env, 2)
            print_avg_survival_month = round(mean_survival_month, 2)
            print("Updates : {} \t\t Survival : {}  \t\t Treated : {} \t\t Reward : {}".format(i_update,
                                                                                          print_avg_survival_month, int(treated_months/num_env),
                                                                                          print_avg_reward))
        if i_update % eval_interval == 0:
            # rewards, survivalMonth, actions, states, colors = evaluate(test_env, ppo_agent.policy_old, eval_times)
            g_rewards, _, g_survivalMonth, g_actions, g_states, g_colors = greedy_evaluate(test_env, init_date.item(), ppo_agent.policy_old,
                                                                                        eval_times)
            # writer.add_scalar("Reward/evaluate", rewards, i_update)
            writer.add_scalar("Reward/greedy_evaluate", g_rewards, i_update)
            writer.add_scalar("Survival month", g_survivalMonth, i_update)
            # log_f.write('{},{}\n'.format(actions, g_actions))
            # log_f.flush()
            if g_rewards > reward_record and i_update >= model_save_start_updating_steps:
                reward_record = g_rewards
                colors = treated_colors.insert(-1, g_colors)
                all_states = np.concatenate((np.array(treated_states).reshape(-1, 3), g_states))
                x = np.arange(int(fixed_init_date / 28), all_states.shape[0] + int(fixed_init_date / 28))
                ad = all_states[:, 0]
                ai = all_states[:, 1]
                psa = all_states[:, 2]
                fig = plt.figure(figsize=(15, 5))
                ax1 = fig.add_subplot(1, 3, 1)
                ax1.plot(x, psa, color="black", linestyle="-", linewidth=1)
                plt.scatter(x, psa, color=colors)
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
                x = np.arange(int(fixed_init_date/28), len(treated_states) + g_states.shape[0] + int(fixed_init_date/28))
                all_states = np.concatenate((np.array(treated_states).reshape(-1, 3), g_states))
                colors = treated_colors.insert(-1, g_colors)
                ad = all_states[:, 0]
                ai = all_states[:, 1]
                psa = all_states[:, 2]
                fig = plt.figure(figsize=(15, 5))
                ax1 = fig.add_subplot(1, 3, 1)
                ax1.plot(x, psa, color="black", linestyle="-", linewidth=1)
                plt.scatter(x, psa, color=colors)
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
        if len(survival_deque) == survival_deque.maxlen and (sum(np.array(survival_deque) >= survival_record) >= .95 * survival_deque.maxlen or i_update - online_updates_epoch > 2e3):
            online_updates_epoch = i_update
            # survival_record -= predict_term
            reward_record = -100000
            rewards_deque.clear()
            survival_deque.clear()
            act_list = []
            online_training_results = []
            for i, online_env in enumerate(online_envs):
                print('===== Updating env.{} ====='.format(i))
                _, states = real_envs.reset(init_date.item())
                # get the real patient's psa level by greedy policy
                g_rewards, g_rewards_list, g_survivalMonth, g_actions, g_states, g_colors = greedy_evaluate(test_env, init_date.item(), ppo_agent.policy_old, 1)
                # we only need the half year actions
                treated_actions.append(g_actions[:predict_term])
                for r, replay_act in enumerate(g_actions[:predict_term]):
                    dose = real_envs._action_set[replay_act]
                    states = g_states[i]
                    reward = g_rewards_list[i]
                    new_cpa.append(dose[0])
                    new_leu.append(dose[1])
                    new_psa.append(states[2])
                    new_days.append(init_date.item() + 28)
                    treated_rewards += reward
                    treated_months += 1
                    treated_colors.append('black')
                    treated_states.append(states)
                    init_date += 28
                    init_state = states
                new_data = {'cpa': new_cpa, 'leu':new_leu, 'psa': new_psa, 'days': new_days}
                print("Treated actions: {}".format(treated_actions))
                online_training_results.append(glv_train_online(clinical_data, glv_dir, tuple_args=online_tuple,online_pars=online_pars, real_pars=real_pars, new_data=new_data, printf=True))
            max_episodes_steps = max_episodes_steps - predict_term
            if max_episodes_steps <= 1:
                print("--------------------------------------------------------------------------------------------")
                print("saving online policy at : " + summary_dir + "/online_treatment_policy.csv")
                pd.DataFrame(treated_actions).to_csv(summary_dir + "/online_treatment_policy.csv")
                print("policy saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")
                break
            real_patient = (
                K, real_pars, init_state, weight, base, m1, m2_ad, m2_ai, drug_decay, drug_length, ad_end_c, ai_end_c,
                sl, max_episodes_steps)
            online_patients = [(
                K, on_pars, init_state, weight, base, m1, m2_ad, m2_ai, drug_decay, drug_length, ad_end_c, ai_end_c,
                sl, max_episodes_steps) for on_pars in online_training_results]
            real_envs = CancerControl(patient=real_patient, t=init_date.item())
            online_envs = [CancerControl(patient=online_p, t=init_date.item()) for online_p in online_patients]
            test_env = CancerControl(patient=real_patient, t=init_date.item())

        # update PPO agent
        if i_update % update_timestep == 0:
            loss = ppo_agent.update(decayflag=args.decayflag, grad_clamp=grad_clamp)
            # log in logging file
            # if i_update % log_freq == 0:
            writer.add_scalar('VLoss', loss, i_update)
            writer.add_scalar("Reward/train", mean_rewards_all_env, i_update)


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
    online_parsdir = "./GLV/analysis-dual-sigmoid/online/model_pars/"
    real_parsdir = "./GLV/analysis-dual-sigmoid/model_pars/"
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
    parser.add_argument("--gpu_cpu", type=str, default="cuda", help="Set device as cuda or cpu or mps")
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
    # parser.add_argument('--type', type=float, default=1)
    parser.add_argument('--seed', type=int, default=0)  # random.randint(0,100000))


    parser.add_argument('--patients_pars', type=tuple, default=(0,))
    parser.add_argument('--patients_train', type=list, default=[])
    parser.add_argument('--number', '-n', type=int, help='Patient No., int type, requested',
                        default=11)  # the only one argument needed to be inputted
    parser.add_argument('--pa_t', type=int, help='No. of the patient to be trained, we train 10 for each',
                        default=1)
    parser.add_argument("--g_mode", type=str, default='online')
    parser.add_argument("--patientNo", type=str, default='patient011')


    parser.add_argument('--num_env', type=int, help='number of environments',
                        default=1)
    parser.add_argument('--max_updates', type=int, help='max number of updating times',
                        default=int(1e5))
    parser.add_argument("--eval_interval", type=int, help="interval to evaluate the policy and plot figures",
                        default=50)
    parser.add_argument('--decayflag', type=bool, default=True, help='lr decay flag')
    parser.add_argument('--model_save_start_updating_steps', type=int, default=500,
                        help="The start steps of saving best model")
    parser.add_argument("--eval_times", type=int, default=10, help='The evaluation time of current policy')

    ppo_args = parser.parse_args()

    # glv_parser = argparse.ArgumentParser(description='Patient arguments')
    # glv_parser.add_argument('--number', '-n', default=11, help='Patient No., int type, requested', type=int)
    # glv_parser.add_argument('--g_mode', '-m', default='online', help='Setting online or offline learning', type=str)
    # glv_parser.add_argument('--t', default=1, type=int)
    # glv_parser.add_argument('--patientNo', default='patient011', help='Patient No., str type, requested', type=str)
    # glv_args = glv_parser.parse_args()

    ending_states = ending_states_all_patients.loc[ppo_args.number]
    ppo_args.m2_ad, ppo_args.m2_ai, ppo_args.ad_end_c, ppo_args.sl = ending_states.m_ad, ending_states.m_ai, ending_states.c_ad, ending_states.sl
    # if args.number == 32:
    #     args.m2_ad = 1.
    # if args.number in [11, 12, 19, 25, 36, 54, 85, 88, 99, 101]: # 52 type = 1
    #     args.type = 2.
    if len(str(ppo_args.number)) == 1:
        patientNo = "patient00" + str(ppo_args.number)
    elif len(str(ppo_args.number)) == 2:
        patientNo = "patient0" + str(ppo_args.number)
    else:
        patientNo = "patient" + str(ppo_args.number)
    # glv_args.patient = patientNo
    real_parslist = os.listdir(real_parsdir + patientNo)
    # clinical_data = pd.read_csv("../data/dataTanaka/Bruchovsky_et_al/" + patientNo + ".txt", header=None)
    alldata = LoadData().Double_Drug()
    clinical_data = alldata[patientNo]
    true_psa = np.array(clinical_data[:, 4])
    true_psa = true_psa[~np.isnan(true_psa)]
    cell_size = 5.236e-10
    mean_v = 5
    Mean_psa = 22.1
    REAL_PARS_LIST = []
    # reading the ode parameters and the initial/terminal states
    for arg in real_parslist:
        if arg.endswith('.csv'):
            real_pars_df = pd.read_csv(real_parsdir + patientNo + '/' + arg)
            _, _, states, pars, best_pars = [np.array(real_pars_df.loc[i, ~np.isnan(real_pars_df.loc[i, :])]) for i in range(5)]
            REAL_PARS_LIST.append(best_pars)
    # Init = np.array([mean_v / Mean_psa * true_psa[0] / cell_size, 1e-4 * K[1], true_psa[0]])
    REAL_PARS_ARR = np.stack(REAL_PARS_LIST)
    real_pars = np.mean(REAL_PARS_ARR, axis=0)
    online_pars_df = pd.read_csv(online_parsdir + patientNo + '/Args_' + str(ppo_args.pa_t) + "-"+patientNo + '.csv')
    initial_date, K, states, _, online_pars = [np.array(online_pars_df.loc[i, ~np.isnan(online_pars_df.loc[i, :])]) for i in range(5)]
    if ppo_args.g_mode == 'offline':
        Init_states = states[:3] # use the original initial states as the initial states for offline learning
    else:
        Init_states = states[3:] # use the ending states of the online training as the initial states for online learning
    ppo_args.patients_pars = (initial_date, K, Init_states, real_pars, online_pars)

    train_online(ppo_args, clinical_data)

