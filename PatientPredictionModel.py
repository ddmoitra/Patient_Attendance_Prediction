# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 15:50:59 2021 PatientPredictionModel.py

@author: Deepto Moitra

THe purpose of this script is to create a model on several patient characteristics and predict whether these patients will be present/show-up for their doctor's appointment

The features matrix of this model include Age, Hipertension, Diabetes, Alcoholism, Handcap and SMS_received. The target vector is a binary output - Sho/No-Show.

A config JSON file contains the folder locations, filetype and feature information referenced by the scripts which can be easily edited by users for their local use

Several algortihms have been tested with different hyperparameters using 5-fold cross validation using the training dataset - Random Forest, Decision Trees and Logistic Regression

Upon finding the best parameters for each model, a pickle file has been saved respectively

All of the models are then evaluated to determine the best accuracy on the test dataset

"""

import numpy as np
import pandas as pd
import joblib
import os
import json
import FileCheck

### For Standardizing and testing ML models
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV          
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from FileCheck import FileChk
from FileCheck import FileType
from FileCheck import FileQlty
from FileCheck import ColChk
from FileCheck import DQChk
from FileCheck import MdlFrm

### Regression Models 
#from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import BaggingRegressor
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.ensemble import StackingRegressor
#from sklearn.ensemble import VotingRegressor
#from sklearn.ensemble import ExtraTreesRegressor


### Classifier Models
#from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

print("""      
Hi there!

I'm Presto, the Patient Prediction Algorithm. 

Others like Deepto also call me by my nickname, Pret

I'm going to try and predict whether your patient is going to show up :)

Let's see what we have here
      """
      )

print("I'm getting your file details")


rf = RandomForestClassifier()
rf_hyperparameters = {
    'n_estimators':[50,100,150],
    'max_depth':[2,4,8,None]
    }


lr = LogisticRegression()
lr_hyperparameters = {
    'C':[0.001,0.01,0.1,1,10,100,1000]
    }

dt = DecisionTreeClassifier()
dt_hyperparameters = {
    'max_depth':[2,3,4,8,None]
    }

### Formats results from 5-fold cross validation and reutrns best hyperparamter for respective model
def ModelParams(results, modl):
    
    print('See all the different parameters I have tried along with the results')
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means,stds,results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean,3), round(std*2,3), params))
    print('The best parameters for {} are: {}\n'.format(str(modl),results.best_params_))
    return results.best_params_

### Executes 5-fold cross validation on various hyperparaemter combinations and saves each model with its respective best parameters as a pickle file
def GridVal(mod,hyp,nfold,x_tr,y_tr):
    nm = type(mod).__name__
    print('Let us get the best paramters for: ', nm )
    print('I am applying {}-fold cross validation'.format(nfold) )
    cv=GridSearchCV(mod,hyp,cv=nfold)
    cv.fit(x_tr,y_tr.values.ravel())
    start_t=time.perf_counter()
    ModelParams(cv, nm)
    end_t=time.perf_counter()
    print('{} took {} seconds'.format(nm,round((end_t-start_t),5)))
    path = os.getcwd()
    joblib.dump(cv.best_estimator_,path+r'\ModelFiles\{}_model.pkl'.format(nm))
    print('I have saved the model file here: ',path+r'\ModelFiles\{}_model.pkl'.format(nm))
    #return MdlParams(cv,nm)

### Applies ML models as per config file on source dataset by splitting in test and train datasets
def ModelMain(dfrm):
    #print(type(dfrm))
    if isinstance(dfrm, pd.DataFrame):
        features = dfrm[FileCheck.cols]
        x_train, x_test, y_train, y_test = train_test_split(features, dfrm['No-show'], random_state=0)
        #models = {}
        mdlcnt = len(FileCheck.mods)
        i=0
        for i in range(0,mdlcnt):
            if i <= mdlcnt:
                if FileCheck.mods[i] == 'RF':
                    print('Model {} is Random Forest'.format(i+1))
                    i+=1
                    GridVal(rf,rf_hyperparameters,5,x_train,y_train)
                elif FileCheck.mods[i] == 'LR':
                    print('Model {} is Logisitic Regression'.format(i+1))
                    i+=1
                    GridVal(lr,lr_hyperparameters,5,x_train,y_train)
                elif FileCheck.mods[i] =='DT':
                    print('Model {} is Decision Tree'.format(i+1))
                    i+=1
                    GridVal(dt,dt_hyperparameters,5,x_train,y_train)
                else:
                    "Model Not Found - {} Are you trying to add a new model?".format(str(FileCheck.mods[i]))
            else:
                print("Can't do it")
        ModelTest(x_test,y_test)
    else:
        print("Something went wrong with dataframe")
        return False

### Reads in pickle model file and executes model evaluation
def ModelTest(x_tst,y_tst):
    models = {}
    for mdl in FileCheck.modfull:
        #print('Reading Model'+'\{}_model.pkl'.format(mdl))
        models[mdl] = joblib.load(FileCheck.modloc+'\{}_model.pkl'.format(mdl))
    print('Here is a comparison of model performance for your review:')
    for nme,modl in models.items():
        ModelEval(nme, modl,x_tst,y_tst)
    
### Outputs accuracy and latency metrics for each ML model applied on unbiased test dataset
def ModelEval(name,model,test_features,test_labels):
    start=time.perf_counter()
    pred=model.predict(test_features)
    end=time.perf_counter()
    accuracy=round(accuracy_score(test_labels,pred),8)
    #precision= round(precision_score(test_labels, pred,average="binary",pos_label="Yes"),8)
    recall=round(recall_score(test_labels,pred,average="binary",pos_label="Yes"),8)
    print('{} -- Accuracy: {} / Recall: {} / Latency: {} seconds'.format(name,accuracy,recall,round((end-start),8)))  #precision,





ModelMain(FileQlty(FileChk(FileCheck.loctn,FileCheck.Filename,FileCheck.Extsn),FileType(FileCheck.obj)))



