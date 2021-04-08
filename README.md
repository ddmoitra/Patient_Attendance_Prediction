### Title

* Patient Reschedule Evaluation Show/No-Show Tool Observations (PRESTO) a.k.a. Patient Resechedule Evaluation Tool (PRET)

### About / Synopsis

* Hello! I am Deepto Moitra and this is my Patient Prediction Machine Learning project known as "PRET"! This project was developed using Python.

* The purpose of this project is to demonstrate and apply the use of Python's Scikit-Learn module on an interesting Kaggle dataset which challenges data scientists to use features in the dataset and perform a binary classification prediction of whether the patient is going to show up to a doctor's appointment or whether a reschedule is likely

* This is certainly useful to know either for regular doctor's appointment or Coid-19 vaccine appointments!

### Detailed Description 

* There are 3 classifiaction ML models used in this project - Random Forest, Logistic Regression and Decision Trees that are applied on the source files using

* The features matrix of this model include Age, Hipertension, Diabetes, Alcoholism, Handcap and SMS_received. The target vector is a binary output - Sho/No-Show.

*A config JSON file contains the folder locations, filetype and feature information referenced by the scripts which can be easily edited by users for their local use

*Several algortihms have been tested with different hyperparameters using 5-fold cross validation using the training dataset - Random Forest, Decision Trees and Logistic Regression

*Upon finding the best parameters for each model, a pickle file has been saved respectively

*All of the models are then evaluated to determine the best accuracy on the unbiased test dataset

### Findings

* Based on model test results, all 3 classification models appear to be able to predict Show-No/Show whether patients are going to attend their doctor's appointment with 80% accuracy on the unbiased test dataset

### Installation / Software Requirements

* The following tools were used in this project:

> *Python 3.7.6
> *Anaconda3
> *Spyder 4.0.1

### Usage

* Use config.json file to paste local folder location directories, the expected columns, the column indexes, the expected column types

* There are 3 Source Files availabel at different volumes of data:
	* Attendance.csv - Full Volume (100,000+ rows)
	* Attendance_2.csv (2000 rows)
	* Attendacne_3.csv (1000 rows)

* Upon setting the configuration, only the PatientPredictionModel.py script needs to be run; the FileCheck.py script is already referenced and leveraged by atientPredictionModel.py as its own containerized module

* If more Scikit-Learn models are needed, the code would need to be updated in PatientPredictionModel.py

### License / Citation

* Python GPU License: https://docs.python.org/3.7/license.html
* Scikit-Learn BSD License: https://scikit-learn.org/stable/about.html#citing-scikit-learn
* Kaggle Dataset: https://www.kaggle.com/joniarroba/noshowappointments

### Support

* This project is a standalone development initiative without any ongoing support

