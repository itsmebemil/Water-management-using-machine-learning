import pandas as pd
import numpy as np
import warnings
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor
import statsmodels.api as sm
from sklearn.model_selection import train_test_split



warnings.filterwarnings("ignore")

db = pd.read_excel("RainfallandWaterLevel.xlsx")
db.replace([np.inf, -np.inf], np.nan, inplace=True)
db.dropna(inplace=True)



db['Date'] = pd.to_datetime(db['Date'])
db.dropna()

sc = StandardScaler()
X = np.array(db['Date']).reshape(-1, 1)
Y = np.array(db['Total_Water_Level']).reshape(-1, 1)

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.25)

X_train_std = sc.fit_transform(X_Train)
X_test_std = sc.transform(X_Test)
gbr_params = {'n_estimators': 1000,
              'max_depth': 3,
              'min_samples_split': 5,
              'learning_rate': 0.03,
              'loss': 'huber'}
gbr = GradientBoostingRegressor(**gbr_params)

model1 = make_pipeline(StandardScaler(), LinearRegression())
model2 = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=1))
estimators = [
    ("Linear Regression", model1),
    ("Random Regressor", model2),
    ("GRB", gbr)
]

final = StackingRegressor(estimators=estimators, final_estimator=RidgeCV())
final.fit(X_Train, Y_Train.ravel())



df = pd.read_excel("WaterLevelPrediction-ML--main/RainfallandWaterLevel.xlsx", parse_dates=True)
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index(['Date'])
df = df['2013-01-01':'2020-12-03'].resample('W').sum()
y = df['Total_Water_Level']
y_to_train = y[:'2019-01-01']  # dataset to train
y_to_val = y['2019-02-02':]  # last X months for test
predict_date = len(y) - len(y[:'2019-02-02'])

def sarima_eva(y, order, seasonal_order, seasonal_period, y_to_test):
    mod = sm.tsa.statespace.SARIMAX(y,
                                    order=order,
                                    seasonal_order=seasonal_order,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    return (results)

fmodel = sarima_eva(y,(0, 1, 1),(1, 1, 1, 52),52,y_to_val)

pickle.dump(fmodel,open('forecastmodel.pkl' , 'wb'))
forecastmodel = pickle.load(open('forecastmodel.pkl','rb'))
#
pickle.dump(final, open('Model.pkl', 'wb'))
model = pickle.load(open('Model.pkl', 'rb'))
