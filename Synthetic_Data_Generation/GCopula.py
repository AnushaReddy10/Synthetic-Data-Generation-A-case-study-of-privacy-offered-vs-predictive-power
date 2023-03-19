 # -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Prashanth Suresh, Naveen Shaji, Pratik Kamat, Amisha Turkel, Anusha Reddy  
# Created Date: 7th March 2023
# version ='1.0'
# requirements = "sdv,pandas,os"
# Improved runtimes with GPU devices. (cuda 11 pytorch) fix , remove pip install of libculas if cuda already setup on device.
# ---------------------------------------------------------------------------
""" Contains a class for generating synthetic Data using SDV
. Derived Work From https://github.com/sdv-dev """ 
# ---------------------------------------------------------------------------


import pandas as pd
import os
from sdv.tabular import GaussianCopula
from sdv.constraints import create_custom_constraint
from sdv.constraints import Inequality
from sdv.constraints import FixedCombinations
from sdv.constraints import ScalarInequality


# Defining Constraints to map Engineering of Features

# WC < C
wc_constraint = Inequality(
  low_column_name='WC',
  high_column_name='C')

# E < C
e_constraint = Inequality(
  low_column_name='E',
  high_column_name='C')

# C <= 19
c_constraint = ScalarInequality(
    column_name='C',
    relation='<=',
   value=19
   )
 
# Actual C=0 when C <19
def is_valid(column_names, data, exclusion_column):
    column_name=column_names[0]
    is_divisible = (data[column_name] % 19 == 0) & (data[column_name]!=0)
    is_excluded = (data[exclusion_column] == 0)
    #is_zero= data[column_name]!=0
    return (is_divisible | is_excluded )
def transform(column_names, data, exclusion_column):
    column_name = column_names[0]
    data[column_name] = data[column_name] / 19
    return data
def reverse_transform(column_names, transformed_data, exclusion_column):
    column_name = column_names[0]
    is_included = (transformed_data[exclusion_column] == 0)
    rounded_data = transformed_data[is_included][column_name].round()
    transformed_data.at[is_included, column_name] = rounded_data
    transformed_data[column_name] *= 19
    return transformed_data
 
    
# Collating all the constraints 
constraint_object = create_custom_constraint(is_valid_fn=is_valid,transform_fn=transform,
                                             reverse_transform_fn=reverse_transform)
ac_constraint = constraint_object(
       column_names=['C'],
    exclusion_column='AC')

constraints=[c_constraint,e_constraint,wc_constraint,ac_constraint]

# Remove in Production when column names are not masked
features = ['I','MT', 'IP', 'SNS', 'RL', 'OS', 'C','AC', 'WC', 'E','T']



# Guassian Copula class with method to generate synthetic data
# Takes both path and dataframe as input ( dataframe for faster reading for multiple iterations)
# Takes INPUTS epochs , batch size ,samples to generate, overwrite,save
# Overwrite True: Overwrite existing dynthetic data
# Save True : Save generated data
# Code emulates sklearn pipelines for comaptibilty 
"""
Usage Example :
from GCopula import GCopula
gcopula = GCopula("Data/data.csv",overwrite=False,save=True)
synth = gcopula.fit_transform(samples=56070)
synth
"""
fd_0 = {'MT':'gamma', 'C':'beta'}

class GCopula:
    
    def __init__(self,data_path,overwrite=False,save=True):
        
        self.save = save
        self.overwrite = overwrite
        
        if isinstance(data_path,str):
            self.data = pd.read_csv(data_path)
            # Remove in Production when column names are not masked
            self.data.columns=features
        else :
            self.data = data_path
        
    def fit(self,fd=None):
        
        if fd == None:
            fd = fd_0
        self.gcopula = GaussianCopula(primary_key='I', constraints=constraints, field_distributions=fd)
        self.synth_path = "synthdata/gcopula__{}__.csv".format(fd)
          
        
    def transform(self,samples=56070):
        
        self.samples = samples
        
        try:
            if os.path.isfile(self.synth_path) and not self.overwrite:
                print("Generated Data Exists")
                print("Loading saved copy :",)
                self.synth = pd.read_csv(self.synth_path)
                return self.synth
            
            else :
                print("Generated data does not exist / overwrite method has been called")
                print("Generating ....") 
                self.gcopula.fit(self.data)
                self.synth = self.gcopula.sample(num_rows=self.samples)
                self.synth = self.synth[features]
                if self.save:
                    self.synth.to_csv(self.synth_path)
                return self.synth

        except Exception as e:
            print("Error : Model Has to be fit before transformation.")
            print(e)
            
    def fit_transform(self,samples=56070,fd=None):
        
        #fitting
        self.fit(fd)
        
        #transforming
        self.transform(samples)
        
        return self.synth
        
           
