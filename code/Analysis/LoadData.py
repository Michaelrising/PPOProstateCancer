#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:11:40 2021

@author: michael
"""

import numpy as np
import os

class LoadData:
    
    data_path = "../data/dataTanaka/Bruchovsky_et_al"
    
    def Patient(self):
        '''
        Read files into the work space

        Parameters
        ----------
        folder : the path
            

        Returns
        -------
        patient_dict : dict
            the patient data stored in a dict

        '''
        folder_path  = os.pardir + self.data_path
        filelist = os.listdir(folder_path)
        patient_dict ={}
        
        for file in filelist:
            
            patientNo = file[7:10]
            lines = np.genfromtxt(folder_path+"/patient"+patientNo+".txt",delimiter=',',dtype='str')
            patient_data = lines
            patient_dict["patient"+patientNo] = patient_data
            
        return patient_dict
    
    
    def Double_Drug(self):
        '''
        Returns
        -------
        patient_dict : dict for the extracted single_drug administrated patient data

        '''
    
        patient_dict = self.Patient()
        Patient_dict = {}
        for patient in list(patient_dict.keys()):
            patient_data = patient_dict[patient]
            CPA = patient_data[:,2]; LEU = patient_data[:,3]
            
            try:
                #Drugd1[np.where(Drugd1 != '')] = [float(strr) for strr in Drugd1[np.where(Drugd1 != '')]]
                CPA[np.where(CPA == '')] = 0; CPA = CPA.astype(float)
                LEU[np.where(LEU == '')] = 0; LEU = LEU.astype(float)
            except: 
                del patient_dict[patient]
                continue
            # for
            
            PatientNo = patient_data[:,0].astype(int); NoCircle = patient_data[:,6].astype(int)
            
            PSAd = patient_data[:,4]; PSAd[np.where(PSAd == '')] = np.nan; PSAd = PSAd.astype(float)
            treatOnOff = patient_data[:,7].astype(int)
            Day = patient_data[:,-1].astype(int)
            
            _Patient_data = np.array([PatientNo, PSAd, CPA, LEU, NoCircle, treatOnOff, Day]).T
            
            Patient_dict[patient] = _Patient_data
            
        return Patient_dict
    
    
    # def Multi_Drug(self):
    #     '''
    #     Returns
    #     -------
    #     patient_dict : dict for the extracted multi_drug administrated patient data

    #     '''
        
    #     patient_dict = self.Patient()
    #     Patient_dict = {}
    #     for patient in list(patient_dict.keys()):
    #         patient_data = patient_dict[patient]
    #          NoCircle = patient_data[:,6].astype(int)
            
    #         ## Find the data where multi-drug have been admintrated to the patient then set it as the seperated dictionary
        
        
        
    #     return Patient_dict
    
    def _Patient_data(self, patientNo):
        '''
        Extract the interested patient_data named patientNo

        Parameters
        ----------
        patientNo : int
            the interested patient number, i.e. 001/012/108

        Returns
        -------
        _patient_dict : dictionary
            Each element of the dict represents a circle of the therapy for this patient, in each element. PSA/CPA/LEU/OnOff/Days
            
        '''
        patient_data = self.Double_Drug()[patientNo]
        #indece = np.where(np.isnan(patient_data[:,1]))
        #delete the colums where PSA is nan
        #patient_data = np.delete(patient_data, indece, axis = 0)
        # set the drug = 0 if no drug is administrated
        patient_data[np.where(np.isnan(patient_data[:,2])),2]  = 0
        patient_data[np.where(np.isnan(patient_data[:,3])),3]  = 0
        # _patient_dict = {}
        # Nocircle = patient_data[:,4].astype(int)
        # unique_no, indece = np.unique(Nocircle, axis=0, return_index=True); indece = np.append(indece, Nocircle.shape[0])
        # for jj in unique_no: Days
        
        #     _patient_dict["Circle"+str(jj)] = patient_data[indece[jj-1]:indece[jj],[1,2,3,5,6]]# PSA/CPA/LEU/Circles/OnOff/
        
        
                                                           
        return patient_data# PatientNo/PSA/CPA/LEU/Circles/OnOff/Days
        
    
    def __init__(self):
        
        self.Double_Drug()
        



# M = LoadData()

# All = M.Double_Drug()

# for jj in All.keys():
#     patientdata = np.round(All[jj], 2)
#     np.savetxt('/Users/michael/OneDrive - City University of Hong Kong/Project/Net Embedding_RL_Cancer/Matlab_mddel/PatientData/'+jj+'.txt', 
#                patientdata, delimiter = ',')
    
    

    
    
            
            
            
            
            
    


            
            