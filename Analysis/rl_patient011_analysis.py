from _utils import *

if __name__ == '__main__':

    print("============================================================================================")

    parsdir = "../Data/model_pars"
    parslist = os.listdir(parsdir)
    patient_pars = {}
    patient_test = []
    patient_train = []
    # reading the ode parameters and the initial/terminal states
    for args in parslist:
        pars_df = pd.read_csv(parsdir + '/' + args)
        patient = args[5:(-4)]
        patient_train.append(patient)
        if patient not in patient_test:
            patient_pars[patient] = pars_df

    env_dict = gym.envs.registration.registry.env_specs.copy()
    for env in env_dict:
        if 'CancerControl-v0' in env:
            print("Remove {} from registry".format(env))
            del gym.envs.registration.registry.env_specs[env]

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join('config', 'sacd.yaml'))
    parser.add_argument('--env_id', type=str, default='gym_cancer:CancerControl-v0')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument("--cuda_cpu", type=str, default="cuda", help="Set device as cuda or cpu")
    parser.add_argument('--m1', type=float, default=0.5)
    parser.add_argument('--m2', type=int, default=12)
    parser.add_argument('--drug_decay', type=float, default=0.75, help="The decay rate for drug penalty")
    parser.add_argument('--seed', type=int, default=0)  # random.randint(0,100000))
    parser.add_argument('--patients_pars', type=dict, default=patient_pars)
    parser.add_argument('--patients_train', type=list, default=patient_train)
    parser.add_argument('--number', '-n', type=int, help='Patient No., int type, requested',
                        default=11)  # the only one argument needed to be inputted
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

    print("============================================================================================")

    device = set_device() if args.cuda_cpu == "cpu" else set_device(args.cuda)
    m1_list = np.linspace(0.1, 0.8, num=8)
    n_list = np.arange(3, 11)
    res_list = []
    dose_list = []
    ###################### m1 #######################
    for m1 in m1_list:
        args.m1 = np.round(m1,1)
        states, doses = test(args)
        res_list.append(states)
        dose_list.append(doses)
    #################### n ########################
    # for n in n_list:
    #     args.m2 = n
    #     states, doses = test(args)
    #     res_list.append(states)
    #     dose_list.append(doses)
    flag_len =0
    for i, res in enumerate(res_list):
        if flag_len < res.shape[0]:
            flag_len = res.shape[0]
    dose_free_per = []
    for i in range(len(m1_list)):
        dose_i = dose_list[i]
        dose_cpa = np.where(dose_i[:,0] == 0)[0].shape[0]/dose_i.shape[0]
        dose_leu = np.where(dose_i[:,1] == 0)[0].shape[0]/dose_i.shape[0]
        dose_free_per.append([dose_cpa, dose_leu])

    ##############################################
    ##################DRAW PLOTS###################
    ###############################################
    # Initialize the figure style
    pars = patient_pars["patient011"]
    K = pars.loc[1, ~np.isnan(pars.loc[1, :])]
    fig = plt.figure(figsize=(20, 10))
    plt.style.use('seaborn')
    plt.style.use(['science', "nature"])
    # create a color palette
    palette = plt.get_cmap('Set1')
    cs = sns.color_palette("Paired")
    # multiple line plot
    num = 0
    x_lim = (flag_len + 1) * 28

    for i, res in enumerate(res_list):
        num += 1
        experts_ai = np.array(pd.read_csv("patient011_AI_evolution_m1_0" +str(num) + ".csv", header=0))
        # Find the right spot on the plot
        plt.subplot(2, 4, num)

        # plot every group, but discrete
        for re in res_list:
            plt.plot(np.arange(0, re.shape[0]) * 28 , re[:, 1], marker='', color='grey', linewidth=0.6, alpha=0.3)

        # Plot the lineplot
        plt.plot(experts_ai[:min(res.shape[0] * 28, experts_ai.shape[0]),0].astype(np.int), experts_ai[:min(res.shape[0] * 28, experts_ai.shape[0]),1],
                 marker='', color = cs[3], lw = 2.4, alpha=0.9, label = "Expert")
        plt.plot(np.arange(0, res.shape[0]) * 28, res[:, 1], marker='', color=cs[1], linewidth=2.4, alpha=0.9,
                 label="RL")

        # Same limits for every chart
        plt.xlim(0, x_lim)
        # plt.ylim(-2, 22)
        plt.yticks(np.arange(0, 3.5, step=0.5) * 1e+9, fontsize=20)
        # plt.yticks(np.arange(0.5, 3, step=1), labels=np.arange(0.5, 3, step=1), fontsize=12)
        plt.xticks(np.arange(0, 2500, 500), labels=np.arange(0, 2500, 500), fontsize=20)
        if num not in [1, 5]:
            plt.tick_params(labelleft=False)
        plt.text(x = 100, y = 2.35e+9, s = "$c_M =$" + " "+ str(np.round(m1_list[i], 1)), fontsize = 22)

        plt.legend(loc = (0., 0.75),fontsize = 22)
        # # Not ticks everywhere
        # if num in range(7):
        #     plt.tick_params(labelbottom='off')
        # if num not in [1, 4, 7]:
        #     plt.tick_params(labelleft='off')

        # Add title
        # plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num))
    plt.savefig("../PPO_Analysis/final/n/patient011_all_AI_evolution.png", dpi = 300)
    plt.show()
    plt.close()
    cs = sns.color_palette("Paired")
    fig, axes = plt.subplots(2, 4,figsize=(24, 10))
    num = 0
    # x_lim = (flag_len + 1) * 28
    for i, doses in enumerate(dose_list):
        num += 1

        # Find the right spot on the plot
        # fig = plt.subplot(2, 4, num)
        # Plot the lineplot
        COLOR_CPA = cs[0]
        COLOR_LEU = cs[1]
        x_dose = np.arange(0, doses.shape[0]) * 28
        ax1 = axes[(num-1)//4, (num-1)%4]
        ax2 = ax1.twinx()
        ax1.plot(x_dose, doses[:, 0], color=COLOR_CPA, label='CPA', lw=1.5)
        # ax1.set_xlabel("Days", fontsize=14)

        ax1.set_xlim(-5, 2800)
        ax1.set_yticks(np.arange(0, 250, 50), np.arange(0, 250, 50), fontsize = 20)
        ax1.tick_params(axis="y", labelcolor=COLOR_CPA)
        ax1.legend(loc=(0.66, 0.9), fontsize=18, facecolor=COLOR_CPA)
        plt.grid(False)
        ax2.plot(x_dose, doses[:, 1], color=COLOR_LEU, label='LEU', lw=1.5, ls='--')

        ax2.tick_params(axis="y", labelcolor=COLOR_LEU)
        ax2.set_yticks(np.arange(0, 22.5, 7.5),np.arange(0, 22.5, 7.5),  fontsize = 20)
        plt.ylim(-.6, 15)
        ax2.legend(loc=(0.66, 0.8), fontsize=18, facecolor=COLOR_LEU)
        # plt.plot(np.arange(0, res.shape[0]) * 28, res[:, 1], marker='', color=palette(num), linewidth=2.4, alpha=0.9)

        # Same limits for every chart
        # plt.xlim(0, x_lim)
        # plt.ylim(-2, 22)
        if num not in [1, 5]:
            ax1.tick_params(labelleft=False)
            plt.yticks(fontsize = 20)
        else:
            ax1.set_ylabel("CPA (mg/ Day)", fontsize=22)
        if num not in [4, 8]:
            ax2.tick_params(labelright = False)
        else:
            ax2.set_ylabel("LEU (ml/ Month)", fontsize=22)
        ax1.set_xticks( np.arange(0,3000, 500), np.arange(0,3000, 500),fontsize = 20)
        ax2.set_xticks(np.arange(0, 3000, 500), np.arange(0, 3000, 500), fontsize=20)
        plt.text(x = 2800 * 0.7 ,y= 0.7 * 15, s="$c_M =$" + " " + str(np.round(m1_list[i], 1)), fontsize=22)
        # # Not ticks everywhere
        # if num in range(7):
        #     plt.tick_params(labelbottom='off')
        # if num not in [1, 4, 7]:
        #     plt.tick_params(labelleft='off')

        # Add title
        # plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num))
    plt.savefig("../PPO_Analysis/final/n/patient011_all_doses_administration.png", dpi=300)
    plt.show()
    plt.close()

    cs = sns.color_palette("Paired")
    # Initialize the figure style
    # Stack figure

    fig = plt.figure(figsize=(20, 10))
    plt.style.use('seaborn')
    plt.style.use(['science', "nature"])
    # create a color palette
    palette = plt.get_cmap('Set1')
    # multiple line plot
    num = 0
    x_lim = (flag_len + 1) * 28
    for i, res in enumerate(res_list):
        num += 1

        # Find the right spot on the plot
        plt.subplot(2, 4, num)

        # Plot the lineplot
        crt_num = res[:, 0] + res[:, 1]
        y_ad = res[:, 0].reshape(1,-1)/K[0]
        y_ai = res[:, 1].reshape(1,-1)/K[1]
        Y = np.concatenate((y_ai, y_ad), axis = 0)
        plt.stackplot(np.arange(0, res.shape[0]) * 28, Y, labels=['HI', 'HD'], colors=cs[:2])
        # plt.plot(np.arange(0, res.shape[0]) * 28, np.log(res[:, 0]), marker='', color=palette(num), linewidth=2, ls = "-", alpha=0.9)
        plt.legend(loc='upper left', fontsize=22)
        # Same limits for every chart
        plt.xlim(0, x_lim)
        # plt.ylim(-0.1, 2)
        # plt.ylim(-2, 22)
        plt.yticks(np.arange(0, 2.5, step=0.5, dtype=np.float), fontsize=20)
        if num not in [1, 5]:
            plt.tick_params(labelleft=False)


        plt.xticks(fontsize=20)
        plt.text(x=200, y=1.15, s="$c_M =$" + " " + str(np.round(m1_list[i], 1)), fontsize=22)
        # # Not ticks everywhere
        # if num in range(7):
        #     plt.tick_params(labelbottom='off')
        # if num not in [1, 4, 7]:
        #     plt.tick_params(labelleft='off')

        # Add title
        # plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num))
    plt.savefig("../PPO_Analysis/final/n/patient011_all_stack_evolution.png", dpi=300)
    plt.show()
    plt.close()

    fig = plt.figure(figsize=(20, 10))
    sps = GridSpec(2, 4)
    plt.style.use('seaborn')
    plt.style.use(['science', "nature"])
    # create a color palette
    palette = plt.get_cmap('Set1')
    # multiple line plot
    num = 0
    x_lim = (flag_len + 1) * 28

    for i, res in enumerate(res_list):
        num += 1
        bax = brokenaxes(ylims=((0, 15), (27, 30)), hspace=.1, subplot_spec=sps[i])
        bax.plot(np.arange(0, res.shape[0]) * 28, res[:, 2], marker='', color=cs[1], linewidth=2, ls = "-", alpha=0.9)
        # Same limits for every chart
        bax.text(x=res.shape[0] * 28 * 0.65, y=28, s="$c_M =$" + " " + str(np.round(m1_list[i], 1)), fontsize=22)
        # bax.tick_params(labelsize = 20)
        plt.xticks(ticks = np.arange(0, (res.shape[0] + 1)* 28, 400), labels = np.arange(0, (res.shape[0] + 1)* 28, 400),fontsize = 18)
        # bax.set_yticks(ticks = np.arange(0, 30, 2), fontsize = 20)

        # # Not ticks everywhere

        if num in [1,5]:
            bax.tick_params(labelleft=True, labelsize = 18)
        else:
            bax.tick_params(labelleft=False)
        bax.tick_params(labelbottom = False)
        # if num not in [1, 4, 7]:
        #     plt.tick_params(labelleft='off')

        # Add title
        # plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num))
    plt.savefig("../PPO_Analysis/final/n/patient011_all_psa_evolution.png", dpi=300)
    plt.show()
    plt.close()

    dose_cpa = []
    dose_leu = []
    dosage_cpa = []
    dosage_leu = []
    for dose in dose_list:
        cpa = dose[:, 0]
        leu = dose[:, 1]
        non_cpa = np.where(cpa == 0)[0].shape[0]
        non_leu = np.where(leu == 0)[0].shape[0]
        dose_cpa.append([non_cpa/cpa.shape[0], 1 - non_cpa/cpa.shape[0]])
        dose_leu.append([non_leu/leu.shape[0], 1 - non_leu/leu.shape[0]])
        dosage_cpa.append([sum(cpa)/cpa.shape[0]])
        dosage_leu.append([sum(leu)/leu.shape[0]])










