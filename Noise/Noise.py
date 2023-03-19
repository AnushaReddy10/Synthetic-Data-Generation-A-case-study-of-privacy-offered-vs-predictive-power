import pandas as pd
import random
import numpy as np
from copy import deepcopy





# Remove in Production when column names are not masked

features = ['I','MT', 'IP', 'SNS', 'RL', 'OS', 'C','AC', 'WC', 'E','T']

numeric = ['MT', 'C', 'AC','WC',
           'E']
ints = ['C', 'AC','WC',
           'E']
categorical = [ 'IP','OS','RL','SNS']




def mutate(x,p,fmap,col):
    
    roulette = random.uniform(0, 1)
    if roulette > p:
        return random.choice(list(set(fmap[col])-set(x)))
    else :
        return x
    
    
def laplace_noise(x,scale_map,noise,col,ints):
    
    y = x + np.random.laplace(loc=0.0, scale=scale_map[col])*noise
    if col in ints:
        return int(y)
    else :
        return y
    
def add_noise(x,c,fmap,scale_map,p,noise,numeric,ints,categorical):
    
    if c in categorical:
        return mutate(x,p,fmap,c)
    elif c in numeric+ints :
        return laplace_noise(x,scale_map,noise,c,ints)
    else :
        return x
    
    

def add_noise_df(df,fmap,scale_map,p,noise,numeric,ints,categorical):
     
    noisy_df = deepcopy(df)
    for c in noisy_df.columns:
        noisy_df[c] = [add_noise(x,c,fmap,scale_map,p,noise,numeric,ints,categorical) for x in noisy_df[c]]
        
    return noisy_df
        

class Noise:
    
    
    def __init__(self):
        
      
        self.numeric = numeric
        self.ints = ints
        self.categorical = categorical

        
            
    def fit(self,real_data_path):
        
        self.path = real_data_path       
        self.df = pd.read_csv(self.path)
        self.df.columns = features
        
        self.fmap = {}
        for col in self.categorical:
            self.fmap[col] = set(self.df[col])
 

        self.scale_map = {}
        self.ds = self.df[numeric].describe()
        for c in self.numeric:
            self.scale_map[c] = self.ds[c]['std']/2**0.5
            
    def transform(self,synth_data_path,p,noise):
        
        self.synth_path = synth_data_path
        self.synth = pd.read_csv(self.synth_path)
        
        self.p = p
        self.noise = noise
        
        self.save_path = self.synth_path.split(".")[0]+"_p_{}_n_{}".format(self.p,self.noise)+".csv"
        
        self.noisy_data = add_noise_df(df=self.df,fmap=self.fmap,scale_map=self.scale_map,p=self.p,
                                       noise=self.noise,numeric=self.numeric,ints=self.ints,categorical=self.categorical)
        
        self.noisy_data.to_csv(self.save_path)
        
    def fit_transform(self,real_data_path,synth_data_path,p,noise):
        
        self.fit(real_data_path)
        self.transform(synth_data_path,p,noise)
        
        
