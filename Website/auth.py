from flask import Blueprint, Flask, render_template,request,flash,redirect,session,url_for
import pandas as pd
import joblib
app = Flask(__name__)
auth= Blueprint('auth',__name__, static_folder='static')
@auth.route('/predict', methods=['POST','GET'])
def predict():
    if request.method=="POST":
        print("helo")
        bp=request.form.get("bp")
        mc=request.form.get("mc")
        nd=request.form.get("nd")
        np=request.form.get("np")
        lb=request.form.get("lb")
        st=request.form.get("st")
        bp_mapping = {'Very High': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        mc_mapping = {'Very High': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        nd_mapping={'2': 0, '3': 1, '4': 2, 'More': 3}
        np_mapping={'2': 0,  '4': 1, 'More': 2}
        lb_mapping={'Small': 0,  'Medium': 1, 'Big': 2}
        st_mapping={'Low':0,'Medium':1,'High':2}
        bp=bp_mapping[bp]
        mc=mc_mapping[mc]
        nd=nd_mapping[nd]
        np=np_mapping[np]
        lb=lb_mapping[lb]
        st=st_mapping[st]
        print(bp,mc,nd,np,lb,st)
        # Create a DataFrame with user input
        input_data = pd.DataFrame({
            'buying': [bp],
            'maintenance': [mc],
            'doors': [nd],
            'persons': [np],
            'lug_boot': [lb],
            'safety': [st]
        })
        model = joblib.load("trained_car_evaluation_model_rf.pkl")
        # Load the original mapping used for one-hot encoding during training
        mapping = joblib.load("one_hot_encoding_mapping.pkl")

        # Apply one-hot encoding to the input data using the original mapping
        input_data_encoded = pd.get_dummies(input_data).reindex(columns=mapping, fill_value=0)

        # Make predictions on the input data
        predictions = model.predict(input_data_encoded)
        print("Predicted Decision:", predictions[0])
        return render_template('result.html',predictios=predictions[0])
    return render_template("base.html")