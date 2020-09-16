import dash as dash
import dash_table
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import os
from app_helpers import download_image, import_image, lat_long, import_annotation
import matplotlib.pyplot as plt

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

#====================================================================
# prerequisites

#--------------------------------------------------------------------
# import dataset
df = pd.read_json('../datasets/waterBodies.json').T
df["lat"] = (df['min_latitude'] + df['max_latitude'])/2
df["lon"] = (df['min_longitude'] + df['max_longitude'])/2
#mapbox token
token = "pk.eyJ1Ijoia2FybHJhZHRrZSIsImEiOiJja2YyZnVvbzUwODJ6MnVxbHU0cDV4YXAxIn0.a3aiSNjy2BOO0WKg40PSsA"

#--------------------------------------------------------------------


#--------------------------------------------------------------------
# map
figure_mapbox = px.scatter_mapbox(
     df, lat="lat", lon="lon", 
     hover_name="name", zoom=1)
figure_mapbox.update_layout(mapbox_style="stamen-terrain", mapbox_accesstoken=token)
#figure_mapbox.update_layout(margin={"r":0,"t":0,"l":0,"b":0})



# go.Figure()
# for i in df.index:
#     longit, latit = lat_long(data_frame=df, slct_lake=i)
    
#     figure_mapbox.add_trace(
#         go.Scattermapbox(
#             fill = "toself", lon = longit, lat = latit, 
#             marker = { 'size': 1, 'color': "orange"}
#             )
#         )
# figure_mapbox.update_layout(
#     mapbox = {
#         'style': "stamen-terrain"
#         },
#     showlegend = False
#     )
#--------------------------------------------------------------------
#====================================================================




#====================================================================
# app laypout
#--------------------------------------------------------------------
app.layout = html.Div(
    children=[
        html.H1(id='some_title', children="water bodies"),

        html.Div(
            dcc.Dropdown(
                id="slct_lake",
                options=[
                    {"label": i+', '+j, "value":k} for i, j, k in zip(df["name"], df["country"], df.index)
                    ],
                placeholder="select a lake",
                 multi=False,
                value=4,
            #style={"width" : "40%", "backgroundColor": "black"}
                )
            ),
        
        html.Div(
            dcc.Graph(id="lake_map", figure=figure_mapbox)
            ),

        html.Div(
            dcc.Graph(id="satellite_image", figure={})
            )
    ]
)
#--------------------------------------------------------------------
#====================================================================



#====================================================================
# interactivity

#--------------------------------------------------------------------
# location on the map
@app.callback(
    Output(component_id="lake_map", component_property="figure"),
    [Input(component_id="slct_lake", component_property="value")]
)
def mapbox_map(slct_lake):
    dff = df.loc[slct_lake, :]

    longit_center = (dff["min_longitude"]+dff["max_longitude"])/2
    latit_center = (dff["min_latitude"]+dff["max_latitude"])/2
    
    
    figure_mapbox.update_layout(
        mapbox = {
            'center': {'lon': longit_center, 'lat': latit_center},
            'zoom': 7},
        showlegend = False)
    
    return figure_mapbox
#--------------------------------------------------------------------


#--------------------------------------------------------------------
# satellite image for selected lake(2019) + mask
@app.callback(
    Output(component_id="satellite_image", component_property="figure"),
    [Input(component_id="slct_lake", component_property="value")]
)
def display_satellite_image(slct_lake):
    # satelite image
    download_image(data_frame=df, slct_lake=slct_lake)
    image = import_image()
    fig = go.Figure()
    fig.add_trace(go.Image(z=image))
    # mask
    X, Y = import_annotation(data_frame=df, slct_lake=slct_lake)
    for x, y in zip(X, Y):
        x = [i/8 for i in x]  #the original picture was 4096x4096 pixel
        y = [i/8 for i in y]  #scaling to 4096/512 = 8
        fig.add_trace(
            go.Scatter(
                x = x, y = y,
                mode='lines', fill='toself', hoverinfo='none',
                fillcolor='red', marker_color='red', opacity=.3
            )
        )
    return fig
#-------------------------------------------------------------------
#====================================================================



if __name__ == '__main__':
    app.run_server(debug=True)