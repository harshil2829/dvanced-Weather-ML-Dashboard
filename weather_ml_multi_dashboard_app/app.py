from flask import Flask, render_template, request
import requests
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

API_KEY = "YOUR_OPENWEATHER_API_KEY"

def fetch_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    
    temps, humidity, wind = [], [], []
    
    for item in data["list"][:10]:
        temps.append(item["main"]["temp"])
        humidity.append(item["main"]["humidity"])
        wind.append(item["wind"]["speed"])
    
    return temps, humidity, wind

def train_model(values):
    X = np.array(range(len(values))).reshape(-1, 1)
    y = np.array(values)
    
    model = LinearRegression()
    model.fit(X, y)
    
    next_day = np.array([[len(values)]])
    prediction = model.predict(next_day)[0]
    
    return round(prediction, 2)

def generate_chart(data, title, filename, ylabel):
    plt.figure()
    plt.plot(data)
    plt.title(title)
    plt.xlabel("Time Index")
    plt.ylabel(ylabel)
    chart_path = os.path.join("static", filename)
    plt.savefig(chart_path)
    plt.close()
    return chart_path

@app.route("/", methods=["GET", "POST"])
def index():
    temp_pred = None
    humidity_pred = None
    wind_pred = None
    charts = {}
    error = None
    
    if request.method == "POST":
        city = request.form["city"]
        
        try:
            temps, humidity, wind = fetch_weather(city)
            
            temp_pred = train_model(temps)
            humidity_pred = train_model(humidity)
            wind_pred = train_model(wind)
            
            charts["temp"] = generate_chart(temps, "Temperature Trend", "temp.png", "Temperature (°C)")
            charts["humidity"] = generate_chart(humidity, "Humidity Trend", "humidity.png", "Humidity (%)")
            charts["wind"] = generate_chart(wind, "Wind Speed Trend", "wind.png", "Wind Speed (m/s)")
            
        except Exception:
            error = "Error fetching data. Check city name or API key."
    
    return render_template("index.html",
                           temp_pred=temp_pred,
                           humidity_pred=humidity_pred,
                           wind_pred=wind_pred,
                           charts=charts,
                           error=error)

if __name__ == "__main__":
    app.run(debug=True)
