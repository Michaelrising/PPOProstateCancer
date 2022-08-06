# PPOProstateCancer
This project applies PPO to individulize the treatment policy for locally advanced prpstate cancer patients. 

Update 17-May-2022: Codes are OK!

Update 6-Aug-2022: Code Ocean Set Up

## Steps in running codes:
### I. Patients tM-GLV model
#### 1.1 run file ./GLV/glv_train_cpu.py with two parameters: n/t; 'n' denotes the patient number; and 't' is setting for randomized training of model. In our paper, we trained model 10 times with randomized samples for patient001. The scripts is: for $i in {1..10}; do python -u './GLV/glv_train_cpu.py' -n 1 --t $i > './GLV/analysis-sigmoid/analysis_sigmoid_1_$i.log' 2>&1 &  done
#### 1.2 After all patients' models are learned we can go to next step learn the RL strategies

### II. PPO
#### 2.1 run file ./train.py as: do nohup python -u  train.py -n 1 --cuda 0  > './PPO_logs/gym_cancer:CancerControl-v0/train1.log' 2>&1 & 
