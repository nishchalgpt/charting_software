import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import csv
import re
import requests
import datetime
import codecs

def get_data_from_google(ticker, exchange, period=60, days=1):
    
    uri = 'http://www.google.com/finance/getprices?x={exchange}&i={period}&p={days}d&f=d,o,h,l,c,v&df=cpct&q={ticker}'.format(ticker=ticker, period=period, days=days, exchange=exchange)
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

app = dash.Dash()

app.layout = html.Div(children=[
    html.Div(children='Symbol to Graph'),
    dcc.Input(id='input', value='', type='text'),
    html.Div(id='output-graph')
    ])

@app.callback(
    Output(component_id='output-graph', component_property='children'),
    [Input(component_id='input', component_property='value')]
    )

def update_graph(input_data):
    
    df = get_data_from_google(input_data, 'NASDAQ', period=60, days=1)
    
    return dcc.Graph(id='example-graph', figure = {
        'data' : [
            {'x': df.index, 'y': df.Close, 'type':'line', 'name':input_data},
            ],
        'layout' : {
            'title': input_data
        }
    })

if __name__ == '__main__':
    app.run_server(debug=True)