from flask import Flask,request, url_for, redirect, render_template
import pickle
import pandas as pd
app = Flask(__name__)

#loading the Random Forest model
model = pickle.load(open("fetal_classification_trained_model.pickle", "rb"))

#Index.html will be returned for the input
@app.route('/')
def hello_world():
    return render_template("index.html")

# predict function, POST method to take in inputs
@app.route('/predict', methods = ['POST','GET'])
def predict(): # take inputs for all the attributes through the HTML form
    FHR_base_value = request.form['1']
    accelerations = request.form['2']
    fetal_movement = request.form['3']
    uterine_contractions = request.form['4']
    light_decelerations = request.form['5']
    severe_decelerations = request.form['6']
    prolongued_decelerations = request.form['7']
    time_STV = request.form['8']
    val_STV = request.form['9']
    time_LTV = request.form['10']
    val_LTV = request.form['11']

    #form a dataframe with the inputs
    test_df = pd.DataFrame([pd.Series([FHR_base_value,accelerations,fetal_movement,uterine_contractions,
                                     light_decelerations,severe_decelerations,prolongued_decelerations,
                                     time_STV,val_STV,time_LTV,val_LTV ]) ])
    print(test_df)

    # predict the class of the fetal health
    prediction = model.predict(test_df)[0]
    if prediction==1.0:
        return render_template('result.html', pred = 'The fetal health is quite normal.'
                                                     '\nNothing to be worried about :)')
    elif prediction==2.0:
        return render_template('result.html', pred='The fetal health seems a bit suspicious.'
                                                   '\nBest would be to supervise it now.')
    else:
        return render_template('result.html', pred='The fetal health is Pathological.\nNeeds Urgent attention!!!')

if __name__ == '__main__':
    app.run(debug=True)
