from flask import Flask, render_template, request, json
from artifects_dir.path_artifect import *
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

app = Flask(__name__)


@app.route('/', methods=['GET'])
def Home():
    return render_template("index.html")


print('initiating prediction')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        fuel = (request.form['fuel'])
        if fuel == "Petrol":
            fuel_Diesel = 0
            fuel_CNG = 0
            fuel_LPG = 0
            fuel_Petrol = 1

        elif fuel == "Diesel":
            fuel_Diesel = 1
            fuel_CNG = 0
            fuel_LPG = 0
            fuel_Petrol = 0

        elif fuel == 'CNG':
            fuel_Diesel = 0
            fuel_CNG = 1
            fuel_LPG = 0
            fuel_Petrol = 0

        elif fuel == 'LPG':
            fuel_Diesel = 0
            fuel_CNG = 0
            fuel_LPG = 1
            fuel_Petrol = 0

        seller_type = (request.form['seller_type'])
        if seller_type == "Dealer":
            seller_type_Dealer = 1
            seller_type_Individual = 0
            seller_type_Trustmark_Dealer = 0

        elif seller_type == "Individual":
            seller_type_Dealer = 0
            seller_type_Individual = 1
            seller_type_Trustmark_Dealer = 0

        elif seller_type == "Trustmark":
            seller_type_Dealer = 0
            seller_type_Individual = 0
            seller_type_Trustmark_Dealer = 1

        transmission = request.form['transmission']
        if transmission == 'Automatic':
            transmission_Automatic = 1
            transmission_Manual = 0

        elif transmission == 'Manual':
            transmission_Automatic = 0
            transmission_Manual = 1

        owner = (request.form['owner'])
        if owner == 'First Owner':
            owner_First_owner = 1
            owner_Fourth_Above_owner = 0
            owner_Second_owner = 0
            owner_Test_drive_car = 0
            owner_Third_owner = 0

        elif owner == 'Fourth & Above Owner':
            owner_First_owner = 0
            owner_Fourth_Above_owner = 1
            owner_Second_owner = 0
            owner_Test_drive_car = 0
            owner_Third_owner = 0

        elif owner == 'Second Owner':
            owner_First_owner = 0
            owner_Fourth_Above_owner = 0
            owner_Second_owner = 1
            owner_Test_drive_car = 0
            owner_Third_owner = 0

        elif owner == 'Test Drive Car':
            owner_First_owner = 0
            owner_Fourth_Above_owner = 0
            owner_Second_owner = 0
            owner_Test_drive_car = 1
            owner_Third_owner = 0

        elif owner == 'Third Owner':
            owner_First_owner = 0
            owner_Fourth_Above_owner = 0
            owner_Second_owner = 0
            owner_Test_drive_car = 0
            owner_Third_owner = 1

        input_data = [{
            'fuel_CNG': float(fuel_CNG),
            'fuel_Diesel': float(fuel_Diesel),
            'fuel_LPG': float(fuel_LPG),
            'fuel_Petrol': float(fuel_Petrol),
            'seller_type_Dealer': float(seller_type_Dealer),
            'seller_type_Individual': float(seller_type_Individual),
            'seller_type_Trustmark Dealer': float(seller_type_Trustmark_Dealer),
            'transmission_Automatic': float(transmission_Automatic),
            'transmission_Manual': float(transmission_Manual),
            'owner_First Owner': float(owner_First_owner),
            'owner_Fourth & Above Owner': float(owner_Fourth_Above_owner),
            'owner_Second Owner': float(owner_Second_owner),
            'owner_Test Drive Car': float(owner_Test_drive_car),
            'owner_Third Owner': float(owner_Third_owner),
            'year': int(request.form['year']),
            'km_driven': float(request.form['km_driven']),
            'seats': int(request.form['seats']),
            'mileage_kmpl': float(request.form['mileage_kmpl']),
            'engine_cc': float(request.form['engine_cc']),
            'max_power_bhp': float(request.form['max_power_bhp']),
            'torque_kgm': float(request.form['torque_kgm']),
            'rpm': float(request.form['rpm']),
        }]

        dataframe = pd.DataFrame(input_data)

        # called best feature selection model and transform test data
        Feature_selection = load_feature_selection_model()
        x_test_selected = Feature_selection.transform(dataframe)

        # scaling test data
        STDS = StandardScaler()
        test_feature = STDS.fit_transform(x_test_selected)
        print(test_feature)

        # Apply Unsupervised Learning (K-means Clustering) on the test data by calling knn model as
        cluster_data = load_cluster_model()
        cluster_assignments_test = cluster_data.fit_predict(test_feature)

        # Step 4: Combine Cluster Assignments with Original Features for testing sets
        X_test_combined = np.column_stack((x_test_selected, cluster_assignments_test))

        # Predict the target variable on the test set by calling xgboost model
        xgboost_model = load_xgboost_model()
        prediction_result = xgboost_model.predict(X_test_combined)

        output = np.round(prediction_result, 2)
        print(prediction_result)
        if output < 0:
            return render_template('final.html', prediction_texts='Sorry u cannot sell')
        else:
            return render_template('final.html', prediction_texts="Your car price is: {}".format(output))
    else:
        return render_template('final.html')


if __name__ == '__main__':
    app.run()