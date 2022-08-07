# PPOProstateCancer
This project applies PPO to individulize the treatment policy for locally advanced prpstate cancer patients. 

Update 17-May-2022: Codes are OK!

Update 6-Aug-2022: Code Ocean Set Up

## Steps in running codes:
### I. Patients tM-GLV model
1.1 run file ./GLV/glv_train_cpu.py with two parameters: n/t; 'n' denotes the patient number; and 't' is setting for randomized training of model. In our paper, we trained model 10 times with randomized samples for patient001. The scripts is: for \\$i in \{1..10\}; do python -u ./GLV/glv_train_cpu.py -n 1 --t \\$i > ./GLV/analysis-sigmoid/analysis_sigmoid_1_\\$i.log 2>\&1 \&  done
 1.2 After all patients' models are learned we can go to next step learn the RL strategies
 *** Note that the training time for each patient is quite long, ranging from few hours to few days, which depends on the initial values and leraning rate. In our training we set is uniformly for all the patients. This parts can be obtained in our paper. ***

### II. PPO
 2.1 run file ./train.py for patient001 as: do nohup python -u train.py -n 1  > ./PPO_logs/gym_cancer:CancerControl-v0/train1.log 2>&1 & 

 ### III. Analysis
 #### III.1 Generating plots of evolution dynamics 
 3.1 For obtaining patients' plots of evolution dynamics with experts strategies, run scripts: ./Analysis/experts_evolution_plots.py(This can generate pictures like Fig2.a/b Fig4/6.c)

 3.2 For obtaining patients' plots of evolution dynamics with ppo strategies, run scripts: ./Analysis/ppo_evolution_plots.py(This can generate pictures like Fig4/6.b)

#### III.2 Generating Fig2.c/d
 3.3 Run scripts ./Analysis/section2_1_model_analysis.py. The output pics saved as Fig2.c: "./Analysis/Figure/all_pars_distribution.eps" and Fig2.d: "./Analysis/Figure/Validation_PSA.eps"

 #### III.3 Generating Fig.3 
3.4 Run scripts ./Analysis/section2_2_model_analysis.py. The output pics saved as Fig3.a: "./Analysis/Figure/distribution_gamma.eps" and Fig3.b: "./Analysis/Figure/A21_changes.eps" Fig3.c: "./Analysis/Figure/ROC_competition_index.eps"

#### III.4 Generating Fig.4
3.5 