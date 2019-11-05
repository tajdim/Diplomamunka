'''
Python: 3.6
How to start?:
Windows:
SET FLASK_APP=Diplomamunka 
SET FLAK_ENV=development
Linux:
export FLASK_APP=Diplomamunka
export FLASK_ENV=development

FLASK RUN --host=0.0.0.0
Reach:
From local network local ip
From outside network i

First run:
predict page,
then picturelabeling
'''
from flask import Flask, render_template, request, Response, redirect, session
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.datasets import cifar10
import keras
from keras.models import load_model
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from PIL import Image
import operator
from random import randint
from keras import backend as K
import pickle
import sqlite3
import db


PIC_FOLDER = os.path.join('static', 'pictures')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PIC_FOLDER
app.secret_key = 'secret key'
app.config.from_mapping(
        # store the database in the instance folder
        DATABASE='Diplomamunka.sqlite3'
    )
# register the database commands
db.init_app(app)


'''Write the object to a file'''
def writeObject(obj, name):
    with open(name, 'wb') as fp:
        pickle.dump(obj, fp)


'''Read the file from the folder'''
def readObject(name):
    with open (name, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


@app.route("/")
def hello():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    writeObject(x_train, 'x_train')
    writeObject(x_test, 'x_test')
    writeObject(y_train, 'y_train')
    writeObject(y_test, 'y_test')

    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'my.png')
    return render_template('index.html', user_image=full_filename)


@app.route("/predict", methods=['GET', 'POST'])
def predict():
        try:
            #data = request.get_json()
            #years_of_experience = float(data["yearsOfExperience"])

            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            model = load_model('./6layer_model.h5')
            y_test_onehot = to_categorical(y_test)
            pred = model.predict(x_test)
            #writing the pred, y_test, x_test value into file to save it
            writeObject(pred, 'pred_out')
            writeObject(y_test, 'y_test')
            writeObject(x_test, 'x_test')
            writeObject(y_train, 'y_train')
            writeObject(x_train, 'x_train')
            
            loss, acc = model.evaluate(x_test, y_test_onehot)
            print(acc)
            img = Image.fromarray(x_train[0], 'RGB')
            img.save('my.png')
        except ValueError:
            return ("Please enter a number.")

        return render_template('index.html', acc=acc, loss=loss);

@app.route("/makemodel")
def makemodel():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    makemodel_function(x_train, y_train, x_test, y_test)
    return "ok"


# Calculating variance and sorted from the lowwest
def sortingByVariance(pred):
    dic = {}
    for i in range(0, 5999):
        elem = pred[i]
        vari = 0
        for j in elem:
            vari += (0.1-j)**2
        dic[i] = vari
    dic_sorted = dict(sorted(dic.items(), key=operator.itemgetter(1), reverse=False))
    with open('dic_sorted_var', 'wb') as fp:
        pickle.dump(dic_sorted, fp)
    return dic_sorted


# Calculating every element max pred and sorted from the lowwest
def soritngByMaxLowPred(pred):
    dic = {}
    for i in range(0, 5999):
        dic[i] = pred[i][np.argmax(pred[i])]
    dic_sorted = dict(sorted(dic.items(), key=operator.itemgetter(1), reverse=True))
    return dic_sorted


def sortingOriginal(pred, y_test_noisy_onehot):
    dic = {}
    for i in range(0, 5999):
        maxi = np.argmax(pred[i])
        if maxi != np.argmax(y_test_noisy_onehot[i]):
            dic[i] = pred[i][maxi]
    dic_sorted = dict(sorted(dic.items(), key=operator.itemgetter(1), reverse=True))
    return dic_sorted
    

def getfilename(number):
    K.clear_session()

    pred = readObject('pred_out')
    y_test = readObject('y_test')
    x_test = readObject('x_test')

    y_test_onehot = to_categorical(y_test)

    y_test_noisy = np.zeros((6000, 1))
    for i in range(0, 6000):
        y_test_noisy[i] = randint(0, 9)

    y_test_noisy_onehot = to_categorical(y_test_noisy)
    # Here we call the sorting function
    dic_sorted = {}
    if os.path.exists('dic_sorted_var'):
        with open ('dic_sorted_var', 'rb') as fp:
            dic_sorted = pickle.load(fp)
    else:
        dic_sorted = sortingByVariance(pred)
    pic = x_test[list(dic_sorted)[number]] 
    img = Image.fromarray(pic, 'RGB')
    url_filename = "pic"+ str(number) + ".png" 
    img.save('static/pictures/' + url_filename)
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], url_filename)
    number=number+1
    with open('number', 'wb') as fp:
        pickle.dump(number, fp)
    session['number'] = number
    
    return full_filename, dic_sorted[number]


def makemodel_function(x_train, y_train, x_test, y_test):
    try:
        y_train_onehot = to_categorical(y_train)
        y_test_onehot = to_categorical(y_test)
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(units=10, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
                      metrics=['accuracy'])

        # Hyper Parameters
        NR_EPOCH = 20
        BATCH_SIZE = 32

        cnn = model.fit(x_train, y_train_onehot, epochs=NR_EPOCH, batch_size=BATCH_SIZE)

        print("--------------------------------------------")
        loss_and_metrics = model.evaluate(x_test, y_test_onehot)
        print("\n --------------------------------------------")
        print(loss_and_metrics)

        #TODO
        #Save the model


    except ValueError:
        return None


#TODO
def createNewModel():
    #(x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = readObject('x_train')
    y_train = readObject('y_train')
    x_test = readObject('x_test')
    y_test = readObject('y_test')
    

    #TODO I have to add the manually labeled picutres to the train set
    #1. Select the data from the db
    result = db.get_db().execute("SELECT * FROM user_pred_1").fetchall()
    #2. With the ID-s select the picture datas from the x_test
    x_labelled = []
    y_labelled = []
    for row in result:
        x_labelled.append(x_test[row['id']])
        y_labelled.append(row['label'])
    writeObject(x_labelled, 'x_labelled') #TODO
    writeObject(y_labelled, 'y_labelled')
    #3. Add the picture datas to he x_train and the label to the y_train
    x_train_new = np.append(x_train,x_labelled)
    y_train_new = np.append(y_train,y_labelled)

    makemodel_function(x_train_new, y_train_new, x_test, y_test)



def writeToDB(label, number):
    with open ('dic_sorted_var', 'rb') as fp:
        dic_sorted = pickle.load(fp)
    pic_id = list(dic_sorted)[number]

    db.get_db().execute(f"INSERT INTO user_pred_1 (id, label) VALUES ('{pic_id}', '{label}')")


@app.route("/picturelabeling", methods=['GET', 'POST'])
def pictureLabeling():
    number = 0
    if not os.path.exists('number'):
        with open('number', 'wb') as fp:
            pickle.dump(number, fp)
    else:
        with open ('number', 'rb') as fp:
            number = pickle.load(fp)
    full_filename, pred_value = getfilename(number)
    return render_template('pic_label.html', user_image=full_filename, pred_value=pred_value)

@app.route("/addlabel", methods=['GET', 'POST'])
def addLabel():
    if request.method == 'POST':
        label = request.form["PicLabel"]
        number = session.get('number')
        writeToDB(label, number)
        full_filename, pred_value = getfilename(number)
        print(number)
        if number%10 == 0:
            createNewModel()
    return render_template('pic_label.html', user_image=full_filename, pred_value=pred_value)


if __name__ == '__main__':
    #app.secret_key = 'A0Zr98j/3yX R~XHH!jmN]LWX/,?RT'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(host = '0.0.0.0',port=5000,debug=True)