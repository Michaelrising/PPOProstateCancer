# PPOProstateCancer
This project applies PPO to individulize the treatment policy for locally advanced prpstate cancer patients. 

Update 17-May-2022: Codes are OK!

Update 6-Aug-2022: Code Ocean Set Up

## Steps in running codes:
### I. Patients tM-GLV model
1.1 run file ./GLV/glv_train_cpu.py with two parameters: n/t; 'n' denotes the patient number; and 't' is setting for randomized training of model. In our paper, we trained model 10 times with randomized samples for patient001. The scripts is: for \\$i in \{1..10\}; do python -u ./GLV/glv_train_cpu.py -n 1 --t \\$i > ./GLV/analysis-sigmoid/analysis_sigmoid_1_\\$i.log 2>\&1 \&  done
 1.2 After all patients' models are learned we can go to next step learn the RL strategies

 ** Note 1 that the training time for each patient is quite long, ranging from few hours to few days, which depends on the initial values and leraning rate. In our training we set is uniformly for all the patients. This parts can be obtained in our paper. ** 

### II. PPO
 2.1 run file ./train.py for patient001 as: do nohup python -u train.py -n 1  > ./PPO_logs/gym_cancer:CancerControl-v0/train1.log 2>&1 & 

** Note 2 that the runing time for PPO is ranging from 1 to few hours. **

** Note 3 Due to the computational-comsuming runing time, we have already collected good outputs for each patiants saved in location: ./PPO_protrained/analysis. And can just use them to reproduce all of the results. If you wanna reproduce your own results, pls note that the path needed to be changed to desired trained model.**
 ### III. Analysis
 #### III.1 Generating plots of evolution dynamics 
 3.1 For obtaining patients' plots of evolution dynamics with experts strategies, run scripts: ./Analysis/experts_evolution_plots.py(This can generate pictures like Fig2.a/b Fig4/6.c)

 3.2 For obtaining patients' plots of evolution dynamics with ppo strategies, run scripts: ./Analysis/ppo_evolution_plots.py(This can generate pictures like Fig4/6.b)

#### III.2 Generating Fig2.c/d
 3.3 Run scripts ./Analysis/section2_1_model_analysis.py. The output pics saved as Fig2.c: "./Analysis/Figure/all_pars_distribution.eps" and Fig2.d: "./Analysis/Figure/Validation_PSA.eps"

 #### III.3 Generating Fig.3 
3.4 Run scripts ./Analysis/section2_2_model_analysis.py. The output pics saved as Fig3.a: "./Analysis/Figure/distribution_gamma.eps" and Fig3.b: "./Analysis/Figure/A21_changes.eps" Fig3.c: "./Analysis/Figure/ROC_competition_index.eps"

#### III.4 Generating Fig.4/6
3.5 Run scripts ./Analysis/section2_3_1_resistanse_dosing_analysis.py or section2_3_1_response_dosing_analysis.py to obtain fig4/6.a. b/c is obtained in steo III.1

#### III.5 Generating Fig5/7
3.6 Run scripts ./Analysis/section2_3_competition_analysis.py to obtain Fig5/7.a 

3.7 Run scripts ./Analysis/section2_3_TTP_analysis.py to obtain Fig5/7. b&c
