import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, Event
import plotly
import plotly.graph_objs as go
import pandas as pd
import csv
import re
import requests
import datetime
import codecs
import numpy as np
import math
from sklearn import preprocessing, cross_validation
from sklearn import svm
import quandl as qd
from collections import deque

def get_data_from_google(ticker, exchange, period=86400, years=1):
    
    uri = 'http://www.google.com/finance/getprices?x={exchange}&i={period}&p={years}Y&f=d,o,h,l,c,v&df=cpct&q={ticker}'.format(ticker=ticker, period=period, years=years, exchange=exchange)
    page = requests.get(uri)
    reader = csv.reader(codecs.iterdecode(page.content.splitlines(), "utf-8"))
    columns = ['Close', 'High', 'Low', 'Open', 'Volume']
    rows = []
    times = []
    
    for row in reader:
        if re.match('^[a\d]', row[0]):
            if row[0].startswith('a'):
                start = datetime.datetime.fromtimestamp(int(row[0][1:]))
                times.append(start)
            else:
                times.append(start+datetime.timedelta(seconds=period*int(row[0])))
            rows.append(map(float, row[1:]))
    
    if len(rows):
        return pd.DataFrame(rows, index=pd.DatetimeIndex(times, name='Date'), columns=columns)
    else:
        return pd.DataFrame(rows, index=pd.DatetimeIndex(times, name='Date'))

def RSI(dataframe, period):
    delta = dataframe.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
    d = d.drop(d.index[:(period-1)])
    rs = pd.stats.moments.ewma(u, com=period-1, adjust=False) / \
         pd.stats.moments.ewma(d, com=period-1, adjust=False)
    return 100 - 100 / (1 + rs)

x = []
y = []
X_graph = deque(maxlen=8)
Y_graph = deque(maxlen=8)

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='live-graph', animate=True),
    dcc.Interval(id='graph-update', interval=1000)
    ])

@app.callback(Output('live-graph', 'figure'), events=[Event('graph-update', 'interval')])

def update_graph():
    
    global X_graph
    global Y_graph
    global x
    global y
    
    df = qd.get("NSE/POWERGRID", authtoken="gmWv5h5b4KEUxUbiSneX")
    df.dropna(inplace=True)
    df_temp = df[['Open', 'High', 'Low', 'Close', 'Total Trade Quantity']]
    df_temp['HLP'] = (df_temp['High'] - df_temp['Close']) / df_temp['Close'] * 100.0
    df_temp['Change'] = (df_temp['Close'] - df_temp['Open']) / df_temp['Open'] * 100
    df_temp['EPS'] = (df['Turnover (Lacs)']*100000)/df['Total Trade Quantity']
    df_temp['PE'] = df_temp['Close']/df_temp['EPS']
    df_temp['EWMA12']= pd.ewma(df_temp['Close'], span=12)
    df_temp['EWMA26']= pd.ewma(df_temp['Close'], span=26)
    df_temp['MACD']= (df_temp['EWMA12'] - df_temp['EWMA26'])
    df_temp['RSI']= RSI(df_temp['Close'],6)
    df_use = df_temp[['Close', 'HLP', 'Change', 'Total Trade Quantity', 'EPS', 'PE', 'EWMA12','EWMA26','MACD', 'RSI']]
    forecast_col = 'Close'
    forecast_out = 8
    df_use['Label'] = df_use[forecast_col].shift(-forecast_out)
    df_pred = df_use[-8:]
    df_use.dropna(inplace=True)
    X = df_use.drop(['Label'], 1)
    Y = df_use['Label']
    y_norm = preprocessing.scale(Y)
    x_norm = preprocessing.scale(X.values)
    rows=x_norm.shape[0]
    a=int(rows*0.80)
    X_train=x_norm[:a]
    Y_train=y_norm[:a]
    X_test=x_norm[a:]
    Y_test=y_norm[a:]
    y_orig = df['Close'][-8:].iloc[:].values
    y_orig_mean = y_orig.mean(axis = 0)
    y_orig_std = y_orig.std(axis = 0)
    clf1 = svm.SVR(kernel = 'rbf', gamma = 0.0001, C = 1000, epsilon = 0.001)
    clf1.fit(X_train, Y_train)
    x_pred = preprocessing.scale(df_pred.drop(['Label'], 1))
    y_pred_clf1  = clf1.predict(x_pred)
    y_new_clf1 = (y_pred_clf1*y_orig_std) + y_orig_mean
    
    Y_graph = y_new_clf1
    
    for i in range(0, 8):
        date = df.index[-1] + datetime.timedelta(days=i)
        X_graph.append(date)
    
    data = go.Scatter(
        x = list(X_graph),
        y = list(Y_graph),
        name = 'Scatter',
        mode = 'lines+markers'
        )
    
    print(X_graph)
    print(Y_graph)
    
    return {'data':[data], 'layout': go.Layout()}

if __name__ == '__main__':
    app.run_server(debug=True)
















