from tkinter import messagebox
from PIL import Image, ImageTk

from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from genetic_selection import GeneticSelectionCV




global filename
global X
le = LabelEncoder()
global mlp_acc, rf_acc, dds_acc
global classifier


def upload():  # function to driving trajectory dataset
    global filename
    filename = filedialog.askopenfilename(initialdir="DrivingDataset")
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n\n")


def generateTrainTestData():
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    train = pd.read_csv(filename)
    train.drop('trajectory_id', axis=1, inplace=True)
    train.drop('start_time', axis=1, inplace=True)
    train.drop('end_time', axis=1, inplace=True)
    print(train)
    train['labels'] = pd.Series(le.fit_transform(train['labels']))
    rows = train.shape[0]  # gives number of row count
    cols = train.shape[1]  # gives number of col count
    features = cols - 1
    print(features)
    X = train.values[:, 0:features]
    Y = train.values[:, features]
    print(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    text.insert(END, "Dataset Length : " + str(len(X)) + "\n")
    text.insert(END, "Splitted Training Length : " + str(len(X_train)) + "\n")
    text.insert(END, "Splitted Test Length : " + str(len(X_test)) + "\n\n")


def prediction(X_test, cls):  # prediction done here
    y_pred = cls.predict(X_test)
    for i in range(len(X_test)):
        print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred, details):
    accuracy = accuracy_score(y_test, y_pred) * 100
    text.insert(END, details + "\n\n")
    text.insert(END, "Accuracy : " + str(accuracy) + "\n\n")
    return accuracy


def runRandomForest():
    global rf_acc
    global classifier
    text.delete('1.0', END)
    rfc = RandomForestClassifier(n_estimators=2, random_state=0)
    rfc.fit(X_train, y_train)
    text.insert(END, "Random Forest Prediction Results\n\n")
    prediction_data = prediction(X_test, rfc)
    random_precision = precision_score(y_test, prediction_data, average='macro') * 100
    random_recall = recall_score(y_test, prediction_data, average='macro') * 100
    random_fmeasure = f1_score(y_test, prediction_data, average='macro') * 100
    rf_acc = accuracy_score(y_test, prediction_data) * 100
    text.insert(END, "Random Forest Precision : " + str(random_precision) + "\n")
    text.insert(END, "Random Forest Recall : " + str(random_recall) + "\n")
    text.insert(END, "Random Forest FMeasure : " + str(random_fmeasure) + "\n")
    text.insert(END, "Random Forest Accuracy : " + str(rf_acc) + "\n\n")
    classifier = rfc


def runMLP():
    global mlp_acc
    text.delete('1.0', END)
    cls = MLPClassifier(random_state=1, max_iter=10)
    cls.fit(X_train, y_train)
    text.insert(END, "Multilayer Perceptron Classifier (MLP) Prediction Results\n\n")
    prediction_data = prediction(X_test, cls)
    mlp_precision = precision_score(y_test, prediction_data, average='macro') * 100
    mlp_recall = recall_score(y_test, prediction_data, average='macro') * 100
    mlp_fmeasure = f1_score(y_test, prediction_data, average='macro') * 100
    mlp_acc = accuracy_score(y_test, prediction_data) * 100
    text.insert(END, "Multilayer Perceptron Precision : " + str(mlp_precision) + "\n")
    text.insert(END, "Multilayer Perceptron Recall : " + str(mlp_recall) + "\n")
    text.insert(END, "Multilayer Perceptron FMeasure : " + str(mlp_fmeasure) + "\n")
    text.insert(END, "Multilayer Perceptron Accuracy : " + str(mlp_acc) + "\n\n")


def runDDS():
    global classifier
    global dds_acc
    dds = RandomForestClassifier(n_estimators=45, random_state=42)
    selector = GeneticSelectionCV(dds,  # algorithm name
                                  cv=5,
                                  verbose=1,
                                  scoring="accuracy",
                                  max_features=5,
                                  n_population=10,  # population
                                  crossover_proba=0.5,  # cross over
                                  mutation_proba=0.2,
                                  n_generations=50,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,  # mutation
                                  tournament_size=3,
                                  n_gen_no_change=5,
                                  caching=True,
                                  n_jobs=1)
    selector = selector.fit(X_train, y_train)
    text.insert(END, "DDS Prediction Results\n\n")
    prediction_data = prediction(X_test, selector)
    dds_precision = precision_score(y_test, prediction_data, average='macro') * 100
    dds_recall = recall_score(y_test, prediction_data, average='macro') * 100
    dds_fmeasure = f1_score(y_test, prediction_data, average='macro') * 100
    dds_acc = accuracy_score(y_test, prediction_data) * 100
    text.insert(END, "DDS Precision : " + str(dds_precision) + "\n")
    text.insert(END, "DDS Recall : " + str(dds_recall) + "\n")
    text.insert(END, "DDS FMeasure : " + str(dds_fmeasure) + "\n")
    text.insert(END, "DDS Accuracy : " + str(dds_acc) + "\n\n")
    classifier = selector


def graph():
    height = [rf_acc, mlp_acc, dds_acc]
    bars = ('Random Forest Accuracy', 'MLP Accuracy', 'DDS with Genetic Algorithm Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()


def predictType():
    filename = filedialog.askopenfilename(initialdir="DrivingDataset")
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n\n")
    test = pd.read_csv(filename)
    test.drop('trajectory_id', axis=1, inplace=True)
    test.drop('start_time', axis=1, inplace=True)
    test.drop('end_time', axis=1, inplace=True)
    cols = test.shape[1]
    test = test.values[:, 0:cols]
    predict = classifier.predict(test)
    print(predict)
    for i in range(len(test)):
        if predict[i] == 0:
            text.insert(END, str(test[i]) + " : Decision Strategy is : Lane Change\n\n")
        if predict[i] == 1:
            text.insert(END, str(test[i]) + " : Decision Strategy is : Speed\n\n")
        if predict[i] == 2:
            text.insert(END, str(test[i]) + " : Decision Strategy is : Steering Angle\n\n")



main = tkinter.Tk()
main.title("Driving Decision Strategy")
main.geometry("1124x740")
#main.config(bg='#1B2631')  # Dark background

bg = PhotoImage(file="bg.png")

# Show image using label
label1 = Label(main, image=bg)
label1.place(x=0, y=0)



# Function to add hover effect on buttons
def on_enter(e):
    e.widget.config(bg="#ccfefe", fg="#ff5a00")  # Light gold

def on_leave(e):
    e.widget.config(bg="#488ba3", fg="white")  # Blue button

# Title Label
font_title = ('Helvetica', 18, 'bold')
title = Label(main, text="A Driving Decision Strategy (DDS) Based on Machine Learning for an Autonomous Vehicle",
              bg="#ccfefe", fg='#1f5369', font=font_title, relief='raised', bd=5)
title.pack(pady=10, fill=X)

# Text Widget for Output
font_text = ('Consolas', 12, 'bold')
text = Text(main, height=20, width=111, font=font_text, bg='#000000',fg="#ffffff", relief='sunken', bd=3)
text.place(x=50, y=80)




# Styling for Buttons
font_button = ('Helvetica', 12, 'bold')
button_style = {'font': font_button, 'bg': '#488ba3', 'fg': 'white',   'width': 30, 'height': 2}

# Buttons with Proper Spacing
uploadButton = Button(main, text="Upload Historical Trajectory Dataset", command=upload, **button_style)
uploadButton.place(x=50, y=500)
uploadButton.bind("<Enter>", on_enter)
uploadButton.bind("<Leave>", on_leave)

trainButton = Button(main, text="Generate Train & Test Model", command=generateTrainTestData, **button_style)
trainButton.place(x=400, y=500)
trainButton.bind("<Enter>", on_enter)
trainButton.bind("<Leave>", on_leave)

rfButton = Button(main, text="Run Random Forest Algorithm", command=runRandomForest, **button_style)
rfButton.place(x=750, y=500)
rfButton.bind("<Enter>", on_enter)
rfButton.bind("<Leave>", on_leave)

mlpButton = Button(main, text="Run MLP Algorithm", command=runMLP, **button_style)
mlpButton.place(x=50, y=580)  # Added spacing for better layout
mlpButton.bind("<Enter>", on_enter)
mlpButton.bind("<Leave>", on_leave)

ddsButton = Button(main, text="Run DDS with Genetic Algorithm", command=runDDS, **button_style)
ddsButton.place(x=400, y=580)  # Placed correctly
ddsButton.bind("<Enter>", on_enter)
ddsButton.bind("<Leave>", on_leave)

predictButton = Button(main, text="Predict DDS Type", command=predictType, **button_style)
predictButton.place(x=400, y=660)  # Placed under DDS button
predictButton.bind("<Enter>", on_enter)
predictButton.bind("<Leave>", on_leave)

graphButton = Button(main, text="Accuracy Comparison Graph", command=graph, **button_style)
graphButton.place(x=750, y=580)  # Properly aligned
graphButton.bind("<Enter>", on_enter)
graphButton.bind("<Leave>", on_leave)

main.mainloop()
