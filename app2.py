import dash as dash
import dash_table
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import os
from preprocessing.app_helpers import download_image, import_image, lat_long
import matplotlib.pyplot as plt



app = dash.Dash(__name__)


#====================================================================
# prerequisites

#--------------------------------------------------------------------
# import dataset
df = pd.read_json('datasets/waterBodies.json').T
#--------------------------------------------------------------------


#--------------------------------------------------------------------
# map
figure_mapbox = go.Figure()
for i in df.index:
    longit, latit = lat_long(data_frame=df, slct_lake=i)
    
    figure_mapbox.add_trace(
        go.Scattermapbox(
            fill = "toself", lon = longit, lat = latit, 
            marker = { 'size': 1, 'color': "orange"}
            )
        )
figure_mapbox.update_layout(
    mapbox = {
        'style': "stamen-terrain"
        },
    showlegend = False
    )
#--------------------------------------------------------------------
#====================================================================




#====================================================================
# app laypout
#--------------------------------------------------------------------
app.layout = html.Div(
    [
        html.H1("water bodies"),
        dcc.Dropdown(
            id="slct_lake",
            options=[
                {"label": i+', '+j, "value":k} for i, j, k in zip(df["name"], df["country"], df.index)
                ],
            placeholder="select a lake",
            multi=False,
            value=4,
            style={"width" : "40%"}
            ),
        dcc.Graph(id="lake_map", figure=figure_mapbox),
        dcc.Graph(id="satellite_image", figure={})
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
# satellite image for selected lake
@app.callback(
    Output(component_id="satellite_image", component_property="figure"),
    [Input(component_id="slct_lake", component_property="value")]
)
def display_satellite_image(slct_lake):
    download_image(data_frame=df, slct_lake=slct_lake)
    image = import_image()
    fig = go.Figure()
    fig.add_trace(go.Image(z=image))
    return fig
#-------------------------------------------------------------------
#====================================================================



if __name__ == '__main__':
    app.run_server(debug=True)