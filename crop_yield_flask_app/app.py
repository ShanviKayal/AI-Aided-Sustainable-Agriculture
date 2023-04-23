from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():

    if request.method == "POST":
        
        clf = joblib.load("regr1.pkl")
        # clf = joblib.load("/home/samuelvara/mysite/regr.pkl")
        Area_used = request.form.get("Area_used")
        Avg_temperature = request.form.get("Avg_temperature")
        Avg_rainfall = request.form.get("Avg_rainfall")
        Pesticide_use = request.form.get("Pesticide_use")
        # weight = request.form.get("weight")
        
        X = pd.DataFrame([[Pesticide_use, Avg_rainfall, Avg_temperature, Area_used]], columns = ['Pesticide_use', 'Avg_rainfall', 'Avg_temperature','Area_used'])
        
        prediction = clf.predict(X)[0]
        
    else:
        prediction = ""
        
        
    return render_template("index.html", output = prediction)

if __name__ == '__main__':
    app.run(debug = True)
