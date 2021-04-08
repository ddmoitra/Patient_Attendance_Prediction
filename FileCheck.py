# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 15:07:05 2021

@author: Deepto Moitra

The purpose of this script is to perform a variety file quality checks on the Source File to ensure it aligns to expectations set in the configuration file

"""

import numpy as np
import pandas as pd
import joblib
import os
import sys
from time import time
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import pathlib
import json


### For Standardizing and testing ML models
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV          
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from time import time
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


curdir=os.getcwd()
jsonloc=curdir+r'\config.json'


    
with open(jsonloc,'r') as myfile:
    data=myfile.read()
    
obj = json.loads(data)
    

    
loctn=obj['TrainingFileLocation']
Filename=obj['TrainTestFile']
Extsn=obj['Extsn']
extsn_lst = obj['ExtsnList']
cols = obj['Features']
colidx = obj['ColIdx']
coldqtyp = obj['ColdDQType']
mods=obj['Models']
modfull=obj['ModelFullName']
modloc=obj['ModelFileLocation']
    

### Checks if file exists
def FileChk(filep,nam, xtsn):
    if os.path.exists(filep) == True:
        print("I found your file directory! Here it is - ", filep)
        if os.path.isfile(filep+'\\'+nam+'.'+xtsn)== True:
            print("And I found your file in it - ", nam+'.'+xtsn)
            return filep+'\\'+nam+'.'+xtsn
        else:
            print("Directory found but file not found!")
            return False
    else:
        print("Neither directory nor file found!")
        return False

### Checks filetype against accepted extensions in config file
def FileType(objct):
    jobj=objct
    Exn=jobj['Extsn']
    exn_lst = jobj['ExtsnList']
    if objct != False:
        if Exn in exn_lst:
            print(Exn+" is a great and useful filetype to use!")
            return Exn
        else:
            print("Sorry, I couldn't quite understand the type file you want me to use. Is it amoungst any of these - "+extsn_lst+"?")
    else:
        print("File not found. Filetype cound not be verified.")
    
### Checks if file columns names are correct, in proper oder and dq check each column
def FileQlty(filedq, ext): 
    if ext == 'csv':
        df1 = pd.read_csv(filedq)
        print("I see your {} data. Let me quickly check the columns".format(ext))
        return ColChk(df1)
    elif ext =='txt':
        df1 = pd.read_csv(filedq, sep ="|")
        print("I see your {} data. Let me quickly check the columns".format(ext))
        return ColChk(df1)
    elif ext =='xlsx':
        df1 = pd.read_excel(filedq)
        print("I see your {} data. Let me quickly check the columns".format(ext))
        return ColChk(df1)
    elif ext == 'xls':
        df1 = pd.read_excel(filedq)
        print("I see your {} data. Let me quickly check the columns".format(ext))
        return ColChk(df1)
    else:
        print("There's something wrong with the file. Care to check the location and filetype again?")
        
### Checks if column structure is as per expected format in the config file
def ColChk(dframe):
    if isinstance(dframe, pd.DataFrame):
        colctr=0
        for n in colidx:
            if dframe.columns[n]==cols[colctr]:
                print("{} matches {}".format(dframe.columns[n],cols[colctr]))
                DQChk(dframe[cols[colctr]].dtype)
                colctr+=1
                if colctr != len(cols):
                    print("Column position is correct")
                else:
                    #return True
                    print("All columns positions match!")
                    return MdlFrm (dframe)
            else:
                print("ERROR: {} did not match {}".format(dframe.columns[n],cols[colctr]))
                return False
    else:
        print("Something went wrong with dataframe")
        return False
        

### Performs DQ Check on each of the dataframe columns to ensure they are expected type as per config file
def DQChk(coltyp):
    cctr = 0
#    for t in coldqtyp:
    if coldqtyp[cctr] == coltyp:
        #print("{} Type matches for {}".format(coltyp, coldqtyp[cctr]))
        cctr+=1
        #if cctr != (len(cols)-1):
        #    print(cctr)
        #else:
        #    print("All Column Type matches")
    else:
        print("ERROR: {} is not the expected type at Column {}".format(coltyp,cols[cctr]))
    
### Returns dataframe object
def MdlFrm(dfrm):
    return dfrm
    #print(type(dfrm))
    #if isinstance(dfrm, pd.DataFrame):
    #    features = dfrm[cols]
    #    x_train, x_test, y_train, y_test = train_test_split(features, dfrm['No-show'], random_state=0)
    #else:
    #    print("Something went wrong with dataframe")
    #    return False


#FileQlty(FileChk(loctn),FileType(FileChk(loctn)))

    
#FileQlty(FileChk(loctn),FileType(FileChk(loctn)))














