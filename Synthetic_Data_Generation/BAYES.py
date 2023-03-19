# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Prashanth Suresh, Naveen Shaji, Pratik Kamat, Amisha Turkel, Anusha Reddy  
# Created Date: 7th March 2023
# version ='1.0'
# requirements = "DataSynthesizer,pandas,os"
# ---------------------------------------------------------------------------
""" Contains a class for generating synthetic Data using Data Synthesizer.
Derived Work From https://github.com/DataResponsibly/DataSynthesizer """ 
# ---------------------------------------------------------------------------
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network
import pandas as pd
import os

# Remove in Production when column names are not masked
features = ['I','MT', 'IP', 'SNS', 'RL', 'OS', 'C','AC', 'WC', 'E','T']

#Defining Attribute types

attribute_to_datatype = {
    'MT': 'Float',
    'IP': 'String',
    'SNS': 'String',
    'RL': 'String',
    'OS': 'String',
    'C': 'Integer',
    'WC': 'Integer',  
    'E': 'Integer',
    'AC':'Integer'
 # 'T': 'Integer'
}

attribute_is_categorical = {
    'MT': False,
    'IP': True,
    'SNS': True,
    'RL': True,
    'OS': True,
    'C': False,
    'WC': False,  
    'E': False,
    'AC':True
 # 'T': True
}




# BAYES class with method to generate synthetic data
# Takes only path as input (dataframe for faster reading for multiple iterations)
# Takes INPUTS epochs , batch size ,samples to generate, overwrite,save
# Overwrite True: Overwrite existing dynthetic data
# Save True : Save generated data
# Code emulates sklearn pipelines for comaptibilty 


"""
Usage Example :

from BAYES import BAYES
bayes = BAYES("Data/data.csv",overwrite=False,save=True)
synth = bayes.fit_transform(epsilon=100,k=1,onesamples=100,zerosamples=5)

"""

class BAYES:
    
    def __init__(self,data_path,overwrite=False,save=True):
        
        
        # Data Processing for Bayes : Splitting T
        if not (os.path.isfile("bayes_temp/data0.csv") and os.path.isfile("bayes_temp/data1.csv")):
            df = pd.read_csv(data_path)
            df.columns = features
            df_zero = df[df['T']==0]
            df_one = df[df['T']==1]
            assert(len(df_zero)>0)
            assert(len(df_one)>0)
            df_zero.to_csv('bayes_temp/data0.csv')
            df_one.to_csv('bayes_temp/data1.csv')
            
        self.one = 'bayes_temp/data1.csv'
        self.zero = 'bayes_temp/data0.csv'
            
        self.save = save
        self.overwrite = overwrite
        
        self.describer0 = DataDescriber()
        self.describer1 = DataDescriber()
        
        self.attribute_is_categorical = attribute_is_categorical
        self.attribute_to_datatype = attribute_to_datatype
        
       
    
    def fit(self,epsilon=100,k=2):
        
        self.synth_path = "synthdata/bayes_k_{}_eps_{}.csv".format(k,epsilon)
        self.t0file = 'synthdata/bayesdata_k_{}_eps_{}_tg_{}.csv'.format(k,epsilon,0)
        self.t1file = 'synthdata/bayesdata_k_{}_eps_{}_tg_{}.csv'.format(k,epsilon,1)
       
        if os.path.isfile(self.synth_path) and not self.overwrite:
            print("Generated Data Exists")
            print("Loading saved copy :",)
            self.synth = pd.read_csv(self.synth_path)
            return False
        
        else :
            
            print("Generated data does not exist / overwrite method has been called")
            print("Generating ....")
     
            #For T 0
            self.describer0.describe_dataset_in_correlated_attribute_mode(
            dataset_file = self.zero,
            epsilon = epsilon,
            k = k,
            attribute_to_datatype = self.attribute_to_datatype,
            attribute_to_is_categorical = self.attribute_is_categorical)
            self.description0 = 'bayes_temp/bayes_k_{}_eps_{}_tg_{}.csv'.format(k,epsilon,0)
            self.describer0.save_dataset_description_to_file(self.description0)

            #For T 1
            self.describer1.describe_dataset_in_correlated_attribute_mode(
            dataset_file = self.one,
            epsilon = epsilon,
            k = k,
            attribute_to_datatype = self.attribute_to_datatype,
            attribute_to_is_categorical = self.attribute_is_categorical)
            self.description1 = 'bayes_temp/bayes_k_{}_eps_{}_tg_{}.csv'.format(k,epsilon,1)
            self.describer1.save_dataset_description_to_file(self.description1)
            
            return True
        
        
    def transform(self,onesamples=100,zerosamples=5):
        
        self.onesamples = onesamples
        self.zerosamples = zerosamples
        
        try :
            
            #For T 0 
            self.generator0 = DataGenerator()
            self.generator0.generate_dataset_in_correlated_attribute_mode(self.zerosamples ,self.description0)
            self.generator0.save_synthetic_data(self.t0file)
            
            #For T 1
            self.generator1 = DataGenerator()
            self.generator1.generate_dataset_in_correlated_attribute_mode(self.onesamples ,self.description1)
            self.generator1.save_synthetic_data(self.t1file)
            
            #joining dataframes
            self.synth = pd.concat([pd.read_csv(self.t0file),pd.read_csv(self.t1file)])
            self.synth = self.synth[features]
           
        except Exception as e:
            
            print("Error : Model Has to be fit before transformation.")
            print(e)
        
                    
            
    def fit_transform(self,epsilon=100,k=2,onesamples=100,zerosamples=5):
        
        #fitting
        if self.fit(epsilon=100,k=2):
            #transforming
            self.transform(onesamples=100,zerosamples=5)
            
        if self.save:
            self.synth.to_csv(self.synth_path)
                
        return self.synth     
       
        
        
           
         
