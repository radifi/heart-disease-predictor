#!/usr/bin/python3.6
# all the imports
from __future__ import print_function
import os
import subprocess
from io import StringIO
from subprocess import call
import PIL
from PIL import Image
#import sqlite3
#import sys
#import shutil
#import time
#import traceback
from flask import Flask, request, session, g, redirect, url_for, abort, \
     render_template, flash
#from django.conf.urls import include

#import statsmodels.api as sm
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle
from sklearn import svm
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import re

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import pylab as plt
#import seaborn
import numpy.random as nprnd
import random
import json
import pickle

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.tree import DecisionTreeRegressor
from IPython.display import Image
import pydotplus
import pydot
import graphviz
from sklearn import tree
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#import importlib.util
#spec = importlib.util.spec_from_file_location("graphviz", "/home/cs3514/.local/lib/python3.5/site-packages/graphviz/files.py")
#foo = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(foo)
#foo.MyClass()

app = Flask(__name__) # create the application instance :)
app.config.from_object(__name__) # load config from this file , flaskr.py

my_dir = os.path.dirname(__file__)
pickle_file_path = os.path.join(my_dir,'CLI_pickle_file.pkl')
with open(pickle_file_path,'rb') as pickle_file:
    decision_tree_CLI = pickle.load(pickle_file)

my_dir_fram = os.path.dirname(__file__)
pickle_file_fram_path = os.path.join(my_dir_fram,'framingham_pickle_file.pkl')
with open(pickle_file_fram_path,'rb') as pickle_file_fram:
    decision_tree_fram = pickle.load(pickle_file_fram)

def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to "
             "produce visualization")

visualize_tree(decision_tree_fram, feature_names=['x'])

features = ['Sex', 'Tot Chol', 'Age', 'Systolic', 'Diastolic', 'Smoker?',
                 'Num Cig','BMI', 'Diabetes', 'BP meds?', 'Heart rate', 'Glucose','Education']
target = ['Low risk', 'High risk']
dot_data = StringIO()
dot_data = export_graphviz(decision_tree_fram, out_file=None,
                        feature_names=features,
                         class_names=target,
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

@app.route('/')
def show_entries():
    return render_template('show_entries.html')

@app.route('/add', methods=['POST'])
def add_entry():
    if request.form['age'] == "" or request.form['gender'] == "" or request.form['Cp'] == "" or \
    request.form['Trestbps'] == "" or request.form['Chol'] =="" or request.form['Fbs'] == "" or \
    request.form['Restecg'] == "" or request.form['Thalach'] == "" or request.form['Exang'] == ""\
    or request.form['Old_Peak_ST'] == "" or request.form['Slope'] == "" or request.form['Ca'] == ""\
    or request.form['Thal'] == "":
        return render_template("failure.html")
    query = [request.form['age'], request.form['gender'], request.form['Cp'], request.form['Trestbps'], request.form['Chol'], request.form['Fbs'], request.form['Restecg'], request.form['Thalach'], request.form['Exang'], request.form['Old_Peak_ST'], request.form['Slope'], request.form['Ca'], request.form['Thal']]
    prediction = decision_tree_CLI.predict(query)
    predict_CLI_prob = decision_tree_CLI.predict_proba(query)
    path_CLI = decision_tree_CLI.decision_path(query)

    prob_score_string_CLI = "{:.3f}".format(np.amax(predict_CLI_prob)*100)
    predict2 = np.array_str(prediction)
    if prediction == 1:
        return render_template('image_display2.html', outcome_clev='You are at risk for heart disease. Probability score of: ' + prob_score_string_CLI)
    else:
        return render_template('image_display2.html', outcome_clev='Congrats! You are not at risk for heart disease. Probability score of: ' + prob_score_string_CLI)


@app.route('/add2', methods=['POST'])
def add_entry2():
    if request.form['SEX'] == "" or request.form['TOTCHOL'] == "" or \
    request.form['AGE'] == "" or request.form['SYSBP'] == "" or \
    request.form['DIABP'] == "" or request.form['CURSMOKE'] == "" or \
    request.form['CIGPDAY'] == "" or request.form['BMI'] == "" or \
    request.form['DIABETES'] == "" or request.form['BPMEDS'] == "" or\
    request.form['HEARTRTE'] == "" or request.form['GLUCOSE'] == "" or\
    request.form['EDUC']:
        return render_template("failure.html")
    query2 = [request.form['SEX'], request.form['TOTCHOL'], request.form['AGE'], request.form['SYSBP'], request.form['DIABP'], request.form['CURSMOKE'], request.form['CIGPDAY'], request.form['BMI'], request.form['DIABETES'], request.form['BPMEDS'], request.form['HEARTRTE'], request.form['GLUCOSE'], request.form['EDUC']]
    predict_fram = decision_tree_fram.predict(query2)
    predict_fram_prob = decision_tree_fram.predict_proba(query2)

    path_fram = decision_tree_fram.decision_path(query)
    prob_score_string_fram = "{:.3f}".format(np.amax(predict_fram_prob)*100)
    query_test2 = np.array_str(predict_fram)

    if predict_fram == 1:
        return render_template('image_display.html', outcome_fram='You are at risk for heart disease. Probability score of: ' + prob_score_string_fram)
    else:
        return render_template('image_display.html', outcome_fram='Congrats! You are not at risk for heart disease. Probability score of: ' + prob_score_string_fram)
