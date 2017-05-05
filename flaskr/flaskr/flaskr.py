
#!/usr/bin/env python
# all the imports
import os
import sqlite3
#import sys
#import shutil
#import time
#import traceback
from flask import Flask, request, session, g, redirect, url_for, abort, \
     render_template, flash
#from django.conf.urls import include
     
import statsmodels.api as sm
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.model_selection import train_test_split
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
import seaborn
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
     
app = Flask(__name__) # create the application instance :)
app.config.from_object(__name__) # load config from this file , flaskr.py

training_data = 'data/HeartData_ALL.csv'
#dependent_variable = include[-1]

# Loading the saved decision tree model pickle
decision_tree_pkl_filename = 'heart_disease_decision_regressor_tree.pkl'
decision_tree_model_pkl = open(decision_tree_pkl_filename, 'rb')
decision_tree_model = pickle.load(decision_tree_model_pkl)
#print ("Loaded Decision tree model :: ", decision_tree_model)

# Load default config and override config from an environment variable
app.config.update(dict(
    DATABASE=os.path.join(app.root_path, 'flaskr.db'),
    DATABASE2=os.path.join(app.root_path, 'patient.db'),
    SECRET_KEY='development key',
    USERNAME='admin',
    PASSWORD='default'
))
app.config.from_envvar('FLASKR_SETTINGS', silent=True)

def connect_db():
    """Connects to the specific database."""
    rv = sqlite3.connect(app.config['DATABASE'])
    rv.row_factory = sqlite3.Row
    return rv
    
def init_db():
    db = get_db()
    with app.open_resource('schema.sql', mode='r') as f:
        db.cursor().executescript(f.read())
    db.commit()

@app.cli.command('initdb')
def initdb_command():
    """Initializes the database."""
    init_db()
    print('Initialized the database.')
    
def get_db():
    """Opens a new database connection if there is none yet for the
    current application context.
    """
    if not hasattr(g, 'sqlite_db'):
        g.sqlite_db = connect_db()
    return g.sqlite_db
    
@app.teardown_appcontext
def close_db(error):
    """Closes the database again at the end of the request."""
    if hasattr(g, 'sqlite_db'):
        g.sqlite_db.close()
        
def connect_db2():
    """Connects to the specific database."""
    rv2 = sqlite3.connect(app.config['DATABASE2'])
    rv2.row_factory = sqlite3.Row
    return rv2
    
def init_db2():
    db2 = get_db2()
    with app.open_resource('schema2.sql', mode='r') as f2:
        db2.cursor().executescript(f2.read())
    db2.commit()

@app.cli.command('initdb2')
def initdb2_command():
    """Initializes the database2."""
    init_db2()
    print('Initialized the database2.')
    
def get_db2():
    """Opens a new database connection if there is none yet for the
    current application context.
    """
    if not hasattr(g, 'sqlite_db2'):
        g.sqlite_db2 = connect_db2()
    return g.sqlite_db2
    
@app.teardown_appcontext
def close_db2(error):
    """Closes the database again at the end of the request."""
    if hasattr(g, 'sqlite_db2'):
        g.sqlite_db2.close()
        
@app.route('/')
def show_entries():
    db = get_db()
    cur = db.execute('select age, gender, Cp, Trestbps, Chol, Fbs, Restecg, Thalach, Exang, Old_Peak_ST, Slope, Ca, Thal from entries order by id desc')
    entries = cur.fetchall()
    db2 = get_db2()
    #raise Exception(db2.execute('select * from patient'))
    cur2 = db2.execute('select SEX, TOTCHOL, AGE, SYSBP, DIABP, CURSMOKE, CIGPDAY, BMI, DIABETES, EDUC, HEARTRTE, GLUCOSE, BPMEDS from patient order by id2 desc')
    patient = cur2.fetchall()
    return render_template('show_entries.html', entries = entries, patient=patient)
    
@app.route('/add', methods=['POST'])
def add_entry():
    if request.form['age'] == "" or request.form['gender'] == "":
        return render_template("failure.html")
    #if not session.get('logged_in'):
        #abort(401)
    db = get_db()
    db.execute('insert into entries (age, gender, Cp, Trestbps, Chol, Fbs, Restecg, Thalach, Exang, Old_Peak_ST, Slope, Ca, Thal) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                 [request.form['age'], request.form['gender'], request.form['Cp'], request.form['Trestbps'], request.form['Chol'], request.form['Fbs'], request.form['Restecg'], request.form['Thalach'], request.form['Exang'], request.form['Old_Peak_ST'], request.form['Slope'], request.form['Ca'], request.form['Thal']])
    user_data = db.commit()
    query = [request.form['age'], request.form['gender'], request.form['Cp'], request.form['Trestbps'], request.form['Chol'], request.form['Fbs'], request.form['Restecg'], request.form['Thalach'], request.form['Exang'], request.form['Old_Peak_ST'], request.form['Slope'], request.form['Ca'], request.form['Thal']]
    prediction = decision_tree_model.predict(query)
    query_test = ''.join(str(e) for e in query)
    predict2 = np.array_str(prediction)
    return '{}'.format(predict2)
    #return '{} {}'.format(query_test, predict2)
    #db.commit()
    #flash('New entry was successfully posted')
    #return redirect(url_for('show_entries'))

@app.route('/add2', methods=['POST'])
def add_entry2():
    if request.form['SEX'] == "":
        return render_template("failure.html")
    #if not session.get('logged_in'):
        #abort(401)
    db2 = get_db2()
    db2.execute('insert into patient (SEX, TOTCHOL, AGE, SYSBP, DIABP, CURSMOKE, CIGPDAY, BMI, DIABETES, EDUC, HEARTRTE, GLUCOSE, BPMEDS) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                 [request.form['SEX'], request.form['TOTCHOL'], request.form['AGE'], request.form['SYSBP'], request.form['DIABP'], request.form['CURSMOKE'], request.form['CIGPDAY'], request.form['BMI'], request.form['DIABETES'], request.form['EDUC'], request.form['HEARTRTE'], request.form['GLUCOSE'], request.form['BPMEDS']])
    user_data2 = db2.commit()
    #print ("hello2")
    query2 = [request.form['SEX'], request.form['TOTCHOL'], request.form['AGE'], request.form['SYSBP'], request.form['DIABP'], request.form['CURSMOKE'], request.form['CIGPDAY'], request.form['BMI'], request.form['DIABETES'], request.form['EDUC'], request.form['HEARTRTE'], request.form['GLUCOSE'], request.form['BPMEDS']]
    query_test2 = ''.join(str(e) for e in query2)
    return '{}'.format(query_test2)
    #print ("hello")
    #db2.execute('insert into patient (SEX, TOTCHOL, AGE, SYSBP, DIABP, CURSMOKE, CIGPDAY, BMI, DIABETES, BPMEDS, HEARTRTE, GLUCOSE, educ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
    #             [request.form['SEX'], request.form['TOTCHOL'], request.form['AGE'], request.form['SYSBP'], request.form['DIABP'], request.form['CURBSMOKE'], request.form['CIGPDAY'], request.form['BMI'], request.form['DIABETES'], request.form['BPMEDS'], request.form['HEARTRTE'], request.form['GLUCOSE'], request.form['educ']])
    #user_data2 = db2.commit()
    #print ("hello2")
    #query2 = [request.form['SEX'], request.form['TOTCHOL'], request.form['AGE'], request.form['SYSBP'], request.form['DIABP'], request.form['CURBSMOKE'], request.form['CIGPDAY'], request.form['BMI'], request.form['DIABETES'], request.form['BPMEDS'], request.form['HEARTRTE'], request.form['GLUCOSE'], request.form['educ']]
    #query_test2 = ''.join(str(e) for e in query2)
    #return '{}'.format(query_test2)
    
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != app.config['USERNAME']:
            error = 'Invalid username'
        elif request.form['password'] != app.config['PASSWORD']:
            error = 'Invalid password'
        else:
            session['logged_in'] = True
            flash('You were logged in')
            return redirect(url_for('show_entries'))
    return render_template('login.html', error=error)
    
#@app.route('/predict', methods=['POST'])
#def predict():	
#	query = [65,1,3.4,249,230,1,2,240,1,2.45,2,2.45,3]
#	prediction = decision_tree_model.predict(query)
#	predict2 = np.array_str(prediction)
#	return predict2

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('You were logged out')
    return redirect(url_for('show_entries'))