
#import pymysql
from flask import Flask, render_template, url_for, flash, redirect, request , session, jsonify
from forms import RegistrationForm, LoginForm
#from model_new import File_Pass
from model import File_Pass

import matplotlib.pyplot as plt
from flask_login import LoginManager, login_user, current_user, logout_user, login_required
#import pickle
import os
import numpy as np
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

filenumber = int(os.listdir('saved_conversations')[-1])
filenumber = filenumber+1
file= open('saved_conversations/'+str(filenumber),"w+")
file.write('bot : Hi There! I am a stock market chatbot. You can begin conversation by typing in a message and pressing enter.\n')
file.close()

app = Flask(__name__)
english_bot = ChatBot('Bot',
                      storage_adapter='chatterbot.storage.SQLStorageAdapter',
                      logic_adapters=[
                         {
                            'import_path': 'chatterbot.logic.BestMatch'
                         },

                      ],
                      trainer='chatterbot.trainers.ListTrainer')
english_bot.set_trainer(ListTrainer)


#conn = pymysql.connect(host='127.0.0.1', user='root', password='root', database='xtipl')
#cur = conn.cursor()
#creating database
import sqlite3 
conn = sqlite3.connect('stock_database')
cur = conn.cursor()
try:
    cur.execute('''CREATE TABLE user (
    id integer Primary key  AUTOINCREMENT,
    name varchar(20),
    email varchar(50),
    password varchar(20))''')
    conn.commit()
except:
    pass

app = Flask(__name__)

app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'




@app.route('/')
#@app.route("/home")
def home():
    return render_template('home.html', title='home')


@app.route("/about")
def about():
    return render_template('about.html', title='About')

@app.route("/register", methods=['GET', 'POST'])
def register():
    conn = sqlite3.connect('stock_database')
    cur = conn.cursor()
    if request.method == 'POST':
        name = request.form['uname']
        email = request.form['email']
        password = request.form['psw']

        cur.execute("insert into user(name,email,password) values ('%s','%s','%s')" % (name, email, password))
        conn.commit()
        # cur.close()
        print('data inserted')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route("/login", methods=['GET', 'POST'])
def login():
    conn = sqlite3.connect('stock_database')
    cur = conn.cursor()
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['psw']
        print('asd')
        count = cur.execute('SELECT * FROM user WHERE email = "%s" AND password = "%s"' % (email, password))
        print(count)
        # conn.commit()
        # cur.close()
        l = len(cur.fetchall())
        if l > 0:
            flash(f'Successfully Logged in')
            return render_template('account.html')
        else:
            print('hello')
            flash(f'Invalid Email and Password!')
    return render_template('login.html')

@app.route("/account")
@login_required
def account():
    return render_template('account.html', title='Account')

@app.route("/onnifty50", methods=['GET', 'POST'])
def onnifty50():
    import csv

    with open('nifty50.csv', newline='') as f:
        result = csv.reader(f)
        header = next(result)
        type(header)

        data = [row for row in result]

    return render_template('onnifty50.html', header=header, data=data)

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    response = str(english_bot.get_response(userText))

    appendfile=os.listdir('saved_conversations')[-1]
    appendfile= open('saved_conversations/'+str(filenumber),"a")
    appendfile.write('user : '+userText+'\n')
    appendfile.write('bot : '+response+'\n')
    appendfile.close()

    return response

@app.route('/chatbot')
def chatbot():
   return render_template('index.html')


@app.route("/prediction", methods=['GET', 'POST'])
def prediction():
    return render_template('prediction.html')


@app.route('/analyse', methods=['POST', 'GET'])
def analyse():
    if request.method == 'POST':
        f = request.files['file']
        name = f.filename
        stock = name.upper()
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        obj = File_Pass(stock,final_features)
        
        
   
   
        #obj = File_Pass(stock)   
        lstm_res =obj.lstm_algo()
        cnn_res = obj.cnn_algo()
        print("LSTM:",lstm_res)
        print("CNN LSTM:",cnn_res)
        return render_template('display.html',prediction_text='Predicted and actual value are plotted on graph',quote=stock,res_lst=lstm_res,res_cnn=cnn_res)

@app.route("/logout")
def logout():
   #session['logged_in'] = False
   return home()



@app.route("/Future_price")
def Future_price():
    if request.method == 'POST':
        company = request.form['company']
    return render_template('Future_price.html')



@app.route("/disp2")

def disp2():
    return render_template('disp2.html', title='Graph')

if __name__ == '__main__':
    app.run(debug=True)
 