import streamlit as st
st.title("My Beautiful App")
import numpy as np
import pandas as pd

rng = np.random.RandomState(1)

# %% samples.py
import urllib.request
import json
# from samples import get_forecast_url
def get_forecast_url():
    url = "https://samples.openweathermap.org"       
    response = urllib.request.urlopen(url)
    html = response.read()
    json_data = json.loads(html)
    return json_data['products']['forecast_5days']['samples'][0]

# %% Weather

# get forecast
url = get_forecast_url()
response = urllib.request.urlopen(url)
html = response.read()
json_data = json.loads(html)
time = [l['dt'] for l in json_data['list']]
temp = [l['main']['temp'] for l in json_data['list']]
df = pd.DataFrame({'time': time,'temp':temp})
df.time = pd.to_datetime(df.time, unit='s')
st.sidebar.subheader("Data")
st.sidebar.dataframe(df)

t = st.slider("timeframe",0,40,30)
X = np.arange(len(temp[0:t])).reshape(-1,1)
y = np.array(temp[0:t])


# st.write(df[0:t])
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

regr_1 = DecisionTreeRegressor(max_depth=4)

regr_2 = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=rng
)

regr_1.fit(X, y)
regr_2.fit(X, y)

y_1 = regr_1.predict(X)
y_2 = regr_2.predict(X)

import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(y=y,
    mode='markers',
    name='Temperatures'))
fig.add_trace(go.Scatter(y=y_1,
    mode='lines',
    name='n_estimators=1'))
fig.add_trace(go.Scatter(y=y_2,
    mode='lines',
    name='n_estimators=100'))
st.plotly_chart(fig)
