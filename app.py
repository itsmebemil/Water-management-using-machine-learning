from flask import Flask, request, render_template, Response, send_file
import pickle
import pandas as pd
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from flask import Flask
import numpy as np


app = Flask(__name__)

Model = pickle.load(open('Model.pkl', 'rb'))
fmodel= pickle.load(open('forecastmodel.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/home')
def hello_world1():
    return render_template("index.html")

@app.route('/fore')
def page2():
    return render_template("forecast.html")

@app.route('/pre')
def page3():
    return render_template("predict.html")


@app.route('/prediction', methods=['POST', 'GET'])
def predict():
    date = request.form['Date']
    # final=[np.array(int_features)]
    d = np.array([date])
    d = pd.to_datetime(d, infer_datetime_format=True)
    # d = d.map(dt.datetime.toordinal)
    val = Model.predict(d.values.reshape(-1, 1))
    return render_template('predict.html', pred=val[0])



plt.rcParams["figure.figsize"] = [12, 6]
plt.rcParams["figure.autolayout"] = True



@app.route('/forecast', methods=['POST', 'GET'])
def plot():
    df = pd.read_excel("RainfallandWaterLevel.xlsx", parse_dates=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index(['Date'])
    df = df['2013-01-01':'2020-12-03'].resample('W').sum()
    y  = df['Total_Water_Level']
    fig= Figure()
    ax = fig.add_subplot(1, 1, 1)
    day= int(request.form['days'])
    pred_uc = fmodel.get_forecast(steps= day)
    pred_ci = pred_uc.conf_int()
    ax.plot(y, label='observed')
    ax.plot(pred_uc.predicted_mean, label='Forecast', color='red')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1],color='red', alpha=.25,edgecolors='black', linewidth=5)
    ax.set_xlabel('Date')
    ax.set_ylabel(y.name)
    img = io.BytesIO()
    FigureCanvas(fig).print_png(img)
    return Response(img.getvalue(), mimetype='image/png')



if __name__ == '__main__':
    app.run(debug=True)
