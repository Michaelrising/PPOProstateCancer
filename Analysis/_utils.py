import json
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
from env.gym_cancer.envs.cancercontrol import CancerControl
# import pybullet_envs
import seaborn as sns
from PPO import PPO
from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib
import matplotlib.colors as mplc
from matplotlib.colors import colorConverter as cC


class AnyObject:
    pass


class AnyObjectHandler:
    def __init__(self, color, alpha = None, _hatch = '///', marker = None):
        self.color = color
        self.hatch = _hatch
        self.maker = marker
        self.alpha = alpha
    def legend_artist(self, legend, orig_handle, fontsize,  handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle([x0, y0], width, height, facecolor=self.color,hatch = self.hatch,alpha = self.alpha,
                                   transform=handlebox.get_transform())

        handlebox.add_artist(patch)
        return patch


class AnyObjectHandler_marker:
    def __init__(self,  marker = None):
        self.marker = marker
    def legend_artist(self, legend, orig_handle, fontsize,  handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        marker = matplotlib.markers.MarkerStyle(marker = self.marker)
        # patch = mpatches.Rectangle([x0, y0], width, height, facecolor=self.color,marker = self.marker,
        #                            transform=handlebox.get_transform())
        handlebox.add_artist(marker)
        return marker

class AnyObjectHandler_line_marker_patch:
    def __init__(self, c_line ,c_bg, c_marker, alpha_bg = 0.3, marker = '*', ls='-'):
        self.c_line = c_line
        self.c_bg = c_bg
        self.c_marker = c_marker
        self.bg_alpha = alpha_bg
        self.marker = marker
        self.ls = ls
    def legend_artist(self,  handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle([x0, y0], width, height, facecolor=self.c_bg, alpha = self.bg_alpha,
                                   transform=handlebox.get_transform())
        line_patch = Line2D([x0+height/2, x0+height/2 + width], [y0+height/2, y0+height/2],ls = self.ls, lw=3,
                            clip_path=patch, marker=self.marker, c=self.c_line, mfc=self.c_marker, mec=self.c_marker, ms=20)
        handlebox.add_artist(line_patch)
        return line_patch


def customSmooth(datadir, smoothdir, weight=0.85, json_name = None, csv_name = None):
    if json_name is not None:
        json_path = datadir + json_name
        with open(json_path, 'rb') as f:
            content = f.read()
            data = json.loads(content)
    elif csv_name:
        csv_path = datadir + csv_name
        data = pd.read_csv(csv_path)
    data = pd.DataFrame(data, columns = ["Wall", "Step","Value"])
    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    save = pd.DataFrame({'Step':data['Step'].values,"Value": data['Value'].values,'SValue':smoothed})
    if not os.path.exists(smoothdir + 'training_analysis_smooth'):
        os.mkdir(smoothdir + 'training_analysis_smooth')
    if json_name is not None:
        save.to_json(smoothdir + 'training_analysis_smooth/'+json_name)
    else:
        save.to_csv(smoothdir + 'training_analysis_smooth/'+csv_name)


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


def plot_figure(data, save_path, m1, m2, par = 0):
    if par:
        save_name = "-m1-" + str(m1) + "-m2-" + str(m2) + "_best_survival_time.png"
    else:
        save_name = "-m1-" + str(m1) + "-m2-" + str(m2) + '_best_reward.png'
    states = data["states"]
    doses = data["doses"]
    x = np.arange(states.shape[0]) * 28
    ad = states[:, 0]
    ai = states[:, 1]
    psa = states[:, 2]
    fig = plt.figure(figsize=(15, 5))
    plt.style.use("seaborn")
    # plt.style.use(['science', "nature"])
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(x, psa, linestyle="-", linewidth=1)
    # plt.scatter(x, psa, color=colors)
    ax1.set_xlabel("Time (Days)")
    ax1.set_ylabel("PSA level (ug/ml)")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(x, ad,  linewidth=1, label="HD")
    ax2.plot(x, ai, linewidth=1, label="HI")
    ax2.set_xlabel("Time (Days)")
    ax2.set_ylabel("Cell counts")
    ax2.legend(loc='upper right')
    #plt.savefig(save_path + "_evolution" + save_name, dpi=300)
    plt.show()
    plt.close()

    COLOR_CPA = "#69b3a2"
    COLOR_LEU = '#FF4500'
    x_dose = np.arange(0, doses.shape[0]) * 28
    fig = plt.figure(figsize=(7.5, 5))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(x_dose, doses[:,0], color=COLOR_CPA, label='CPA', lw=1.5)
    ax1.set_xlabel("Days", fontsize=14)
    ax1.set_ylabel("CPA (mg/ Day)", fontsize=14)
    ax1.set_xlim(-5, 2500)
    ax1.set_yticks(np.arange(0, 250, 50))
    ax1.tick_params(axis="y", labelcolor=COLOR_CPA)
    ax1.legend(loc=(0.8, 0.9), fontsize=14, facecolor=COLOR_CPA)
    plt.grid(False)
    ax2.plot(x_dose, doses[:, 1], color=COLOR_LEU, label='LEU', lw=1.5, ls='--')
    ax2.set_ylabel("LEU (ml/ Month)", fontsize=14)
    ax2.tick_params(axis="y", labelcolor=COLOR_LEU)
    ax2.set_yticks(np.arange(0, 22.5, 7.5))
    plt.ylim(-.6, 15)
    ax2.legend(loc=(0.8, 0.8), fontsize=14, facecolor=COLOR_LEU)
    # plt.savefig(save_path+'_dosages' + save_name, dpi=300)
    plt.show()
    plt.close()
    return


def test(args):
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
    list_df = args.patients_pars[patientNo]
    A, K, states, pars, best_pars = [np.array(list_df.loc[i, ~np.isnan(list_df.loc[i, :])]) for i in range(5)]
    A = A.reshape(2, 2)
    # A = np.array(list_df.loc[0, ~np.isnan(list_df.loc[0, :])]).reshape(2, 2)
    # K = np.array(list_df.loc[1, ~np.isnan(list_df.loc[1, :])])
    # states = np.array(list_df.loc[2, ~np.isnan(list_df.loc[2, :])])
    # pars = np.array(list_df.loc[3, ~np.isnan(list_df.loc[3, :])])
    init_state = states[:3]
    terminate_state = states[3:]

    # default_acts = pd.read_csv("../Model_creation/test-sigmoid/model_actions/" + patientNo + "_actions_seqs.csv")
    # default_acts = np.array(default_acts)
    #
    # default_action = np.array(default_acts[:, 0], dtype=np.int)
    weight = np.ones(2) / 2
    base = 1.15
    m1 = args.m1
    m2 = args.m2
    drug_decay = args.drug_decay
    drug_length = 8

    patient = (A, K, best_pars, init_state, terminate_state, weight, base, m1, m2, drug_decay, drug_length)

    test_env = CancerControl(patient=patient)

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


    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    best_directory = "../PPO_preTrained" + '/' + env_name + '/' + patientNo + "/" + "final/"
    if m1 == 0.5:
        best_directory = "../PPO_preTrained" + '/' + env_name + '/' + patientNo + "/" + "best/"
    best_name = os.listdir(best_directory)
    flag_name = ''
    print( "m1-" + str(m1) + "-m2-" + str(m2))
    for name in best_name:
        if "m1-" + str(m1) + "-m2-" + str(m2) in name:
            flag_name = name
            break

    # tempT = []
    # for name in best_name:
    #     checkpoint_path = best_directory + name
    #     t = os.path.getctime(checkpoint_path)
    #     tempT.append(t)
    # index = tempT.index(max(tempT))
    checkpoint_path = best_directory + flag_name # "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    record_states_high_reward = 0
    record_dose_high_reward = 0
    record_states_high_survival_time = 0
    record_dose_survival_month = 0
    test_running_reward = 0
    total_test_episodes = 20
    ALL_STATES = []
    states = []
    doses = []
    record_reward = -1000
    record_survival_month = 0
    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        ep_survival_month = 0
        fea, state = test_env.reset()
        test_env.seed(np.random.randint(1, 1000000))
        states.append(state)
        while True:
            _, action, _ = ppo_agent.greedy_select_action(fea)
            fea, state, reward, done, infos = test_env.step(action)
            states.append(state)
            doses.append(infos["dose"])
            ep_reward += reward
            ep_survival_month += 1
            if done:
                break
        if record_reward < ep_reward:
            record_reward = ep_reward
            record_states_high_reward = np.vstack(states.copy())
            record_dose_high_reward = np.vstack(doses)
        if record_survival_month < ep_survival_month:
            record_survival_month = ep_survival_month
            record_states_high_survival_time = np.vstack(states.copy())
            record_dose_survival_month = np.vstack(doses)
        ALL_STATES.append(np.vstack(states.copy()))
        states.clear()
        doses.clear()
        test_running_reward += ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))

    test_env.close()
    # maximum rewards
    High_reward = {"states": record_states_high_reward, "doses": record_dose_high_reward}
    High_survival = {"states": record_states_high_survival_time, "doses": record_dose_survival_month}
    savepath = "../PPO_Analysis/" + patientNo
    # plot_figure(High_reward, savepath , m1, m2,0)
    # plot_figure(High_survival, savepath, m1, m2, 1)
    # pd.DataFrame(record_dose_high_reward).to_csv(savepath + "-m1-" + str(m1) + "-m2-" + str(m2)+ "_converge_high_reward_dosages.csv")
    # pd.DataFrame(record_dose_survival_month).to_csv(savepath+ "-m1-" + str(m1) + "-m2-" + str(m2) + "_converge_high_survival_dosages.csv")
    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")

    all_states = [np.append(ALL_STATES[i], -np.ones((record_survival_month + 1 - ALL_STATES[i].shape[0], ALL_STATES[i].shape[1])), axis = 0) for i in range(len(ALL_STATES))]
    all_ad = np.vstack([states[:,0] for states in all_states])
    all_ai =  np.vstack([states[:,1] for states in all_states])
    all_psa = np.vstack([states[:,2] for states in all_states])
    return High_reward['states'], High_reward["doses"]

def _to_rgb(c):
    """
    Convert color *c* to a numpy array of *RGB* handling exeption
    Parameters
    ----------
    c: Matplotlib color
        same as *color* in *colorAlpha_to_rgb*
    output
    ------
    rgbs: list of numpy array
        list of c converted to *RGB* array
    """

    if(getattr(c, '__iter__', False) == False):  #if1: if c is a single element (number of string)
        rgbs = [np.array(cC.to_rgb(c)),]  #list with 1 RGB numpy array

    else:  #if1, else: if is more that one element

        try:   #try1: check if c is numberic or not
            np.array(c) + 1

        except (TypeError, ValueError):  #try1: if not numerics is not (only) RGB or RGBA colors
            #convert the list/tuble/array of colors into a list of numpy arrays of RGB
            rgbs = [np.array( cC.to_rgb(i)) for i in c]

        except Exception as e:  #try1: if any other exception raised
            print("Unexpected error: {}".format(e))
            raise e #raise it

        else:  #try1: if the colors are all numberics

            arrc = np.array(c)  #convert c to a numpy array
            arrcsh = arrc.shape  #shape of the array

            if len(arrcsh)==1:  #if2: if 1D array given
                if(arrcsh[0]==3 or arrcsh[0]==4):  #if3: if RGB or RBGA
                    rgbs = [np.array(cC.to_rgb(c)),]  #list with 1 RGB numpy array
                else:   #if3, else: the color cannot be RBG or RGBA
                    raise ValueError('Invalid rgb arg "{}"'.format(c))
                #end if3
            elif len(arrcsh)==2:  #if2, else: if 2D array
                if(arrcsh[1]==3 or arrcsh[1]==4):  #if4: if RGB or RBGA
                    rgbs = [np.array(cC.to_rgb(i)) for i in c]  #list with RGB numpy array
                else:   #if4, else: the color cannot be RBG or RGBA
                    raise ValueError('Invalid list or array of rgb')
                #end if4
            else:  #if2, else: if more dimention
                raise ValueError('The rgb or rgba values must be contained in a 1D or 2D list or array')
            #end if2
        #end try1
    #end if1

    return rgbs

def _is_number(s):
    """
    Check if *c* is a number (from
    http://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-in-python)
    Parameters
    ----------
    c: variable
    output
    ------
    true if c is a number
    false otherwise
    """
    try:
        float(s) # for int, long and float
    except ValueError:
        return False
    return True

def _check_alpha(alpha, n):
    """
    Check if alpha has one or n elements and if they are numberics and between 0 and 1
    Parameters
    ----------
    alpha: number or list/tuple/numpy array of numbers
        values to check
    output
    ------
    alpha: list of numbers
        if all elements numberics and between 0 and 1
    """
    alpha = np.array(alpha).flatten()  #convert alpha to a flattened array
    if(alpha.size == 1):  #if1: alpha is one element
        if(_is_number(alpha) == False or alpha < 0 or alpha > 1):
            raise ValueError("'alpha' must be a float with value between 0 and 1, included")
        else:
            alpha = [alpha for i in range(n)]  #replicate the alphas len(colors) times
    elif(alpha.size==n):  #if1, else: if alpha is composed of len(colors) elements
        try:  #check if all alphas are numbers
            alpha+1
        except TypeError:
            raise ValueError("All elements of alpha must be a float with value between 0 and 1, included")
        else:
            if((alpha < 0).any() or (alpha > 1).any()):
                raise ValueError("'alpha' must be a float with value between 0 and 1, included")
    else:  #if1, else: if none of the previous cases
        raise ValueError("Alpha must have either one element or as many as 'colors'")
    #end if1
    return alpha

def colorAlpha_to_rgb(colors, alpha, bg='w'):

    colors = _to_rgb(colors)  #convert the color and save in a list of np arrays
    bg = np.array(cC.to_rgb(bg))   #convert the background

    #check if alpha has 1 or len(colors) elements and return a list of len(color) alpha
    alpha = _check_alpha(alpha, len(colors))
    #interpolate between background and color
    rgb = [(1.-a) * bg + a*c for c,a in zip(colors, alpha)]

    return rgb


def cmap(cmap_name, alpha, bg="w", set_under=None, set_over=None,
         set_bad=None, out_cmap_name=None):

    # Read input cmap:
    input_cmap = plt.cm.get_cmap(cmap_name)
    ncolors = input_cmap.N

    position = np.linspace(0, 1, ncolors)
    # Convert RGBA colors from cmap into RGB:
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos in position:
        r, g, b = colorAlpha_to_rgb(input_cmap(pos), alpha, bg)[0]
        cdict['red'  ].append((pos, r, r))
        cdict['green'].append((pos, g, g))
        cdict['blue' ].append((pos, b, b))

    # Set output colormap name:
    if out_cmap_name is None:
        out_cmap_name = cmap_name + "_{0:.1f}".format(alpha)
    # mimic-alpha colormap:
    ma_cmap = mplc.LinearSegmentedColormap(out_cmap_name, cdict, 256)

    # Set mimic-alpha colors for masked and out-of-range values:
    if set_under is not None:
        RGBunder = colorAlpha_to_rgb(set_under, alpha, bg)[0]
        ma_cmap.set_under(RGBunder)
    if set_over is not None:
        RGBover  = colorAlpha_to_rgb(set_over,  alpha, bg)[0]
        ma_cmap.set_over(RGBover)
    if set_bad is not None:
        RGBbad   = colorAlpha_to_rgb(set_bad,   alpha, bg)[0]
        ma_cmap.set_bad(RGBbad)

    return ma_cmap

