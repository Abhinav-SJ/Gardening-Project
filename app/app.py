from flask import Flask, render_template,request,url_for, Markup
import pickle
import requests
import config
import pickle
import io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from utils.fertidict import fertdict


app = Flask(__name__)

def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None

@app.route("/")
def home():
    title = 'GARDENING ASSIST'
    return render_template('index.html', title = title)

@app.route('/crop')
def crop():
    title = 'GARDENING ASSIST - CROP RECOMMENDATION'
    return render_template('croppage.html', title=title)

@app.route('/fert')
def fert():
    title = 'GARDENING ASSIST - FERTILIZER RECOMMENDATION'
    return render_template('fertpage.html')

@app.route('/disease')
def disease():
    title = 'GARDENING ASSIST - DISEASE RECOMMENDATION'
    return render_template('diseasepage.html')




@app.route('/predictcrop', methods = ['POST'])
def predictcrop():
    title = 'GARDENING ASSIST - CROP RESULTS'
    crop_recommendation_model_path = 'C:/Users/hp/Documents/FlaskApp/models/RandomForest.pkl'
    crop_recommendation_model = pickle.load(
        open(crop_recommendation_model_path, 'rb'))

    if request.method == 'POST':
        N = int(request.form['Nitrogen'])
        P = int(request.form['Phosphorus'])
        K = int(request.form['Pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")
        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('cropresult.html', prediction=final_prediction, title=title)
        
       
        else:
            return render_template('try_again.html', title=title)

@app.route('/predictfert', methods=['POST'])
def predictfert():
    cp = request.form['crop']
    N = int(request.form['N'])
    P = int(request.form['P'])
    K = int(request.form['K'])
    
    df = pd.read_csv('C:/Users/hp/Documents/FlaskApp/data/fertilizer_data.csv')
    Nr = df[df['Crop'] == cp]['N'].iloc[0]
    Pr = df[df['Crop'] == cp]['P'].iloc[0]
    Kr = df[df['Crop'] == cp]['K'].iloc[0]

    Nd = Nr - N
    Pd = Pr - P
    Kd = Kr - K

    if(abs(Nd) > abs(Pd)):
        if(abs(Nd) > abs(Kd)):
            max = 'N'
        else:
            max = 'K'
    elif(abs(Kd) > abs(Pd)):
        max = 'K'
    else:
        max = 'P'

    if max == 'N':
        key = 'Nlow' if Nd > 0 else 'Nhigh'
    elif max == 'P':
        key = 'Plow' if Pd > 0 else 'Phigh'
    else:
        key = 'Klow' if Kd >0 else 'khigh'

    response = Markup(str(fertdict[key]))

    return render_template('fertresult.html',res = response)

    
if __name__ == '__main__':
    app.run(debug=True)
