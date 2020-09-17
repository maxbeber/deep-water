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

MAP_STYLE= 'outdoors'
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)

#====================================================================
# prerequisites

#--------------------------------------------------------------------
# import dataset
df = pd.read_json('../datasets/waterBodies.json').T
df["lat"] = (df['min_latitude'] + df['max_latitude'])/2
df["lon"] = (df['min_longitude'] + df['max_longitude'])/2

# Plotly mapbox public token
mapbox_access_token = "pk.eyJ1Ijoia2FybHJhZHRrZSIsImEiOiJja2YyZnVvbzUwODJ6MnVxbHU0cDV4YXAxIn0.a3aiSNjy2BOO0WKg40PSsA"

# map
figure_mapbox = px.scatter_mapbox(
     df,
     lat="lat",
     lon="lon", 
     hover_name="name",
     color_discrete_sequence=["fuchsia"],
     zoom=3,
     height=200)
figure_mapbox.update_layout(mapbox_style=MAP_STYLE, mapbox_accesstoken=mapbox_access_token)
figure_mapbox.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})


#====================================================================
# app laypout
#====================================================================
app.layout = html.Div(
    children=[
        html.Div(
            className="row",
            children=[
                # Column for user controls
                html.Div(
                    className="four columns div-user-controls",
                    children=[
                        html.Img(
                            className="logo", src=app.get_asset_url("dash-logo-new.png")
                        ),
                        html.H2("DEEP WATER"),
                        html.P(
                            """Select a water body and choose the desired year."""
                        ),
                        html.Div(
                            className="row",
                            children=[
                                html.Div(
                                    className="div-for-dropdown",
                                    children=[
                                        # Dropdown for water bodies
                                        dcc.Dropdown(
                                            id="slct_lake",
                                            options=[
                                                {"label": name.capitalize() + ', ' + country.replace('_', ' ').capitalize(), "value":i} for name, country, i in zip(df["name"], df["country"], df.index)
                                            ],
                                            value=14,
                                            placeholder="Select a water body",
                                        )
                                    ],
                                ),
                                html.Div(
                                    className="div-for-dropdown",
                                    children=[
                                        # Dropdown to select desired year
                                        dcc.Dropdown(
                                            id="bar-selector",
                                            options=[
                                                {
                                                    "label": str(n),
                                                    "value": str(n),
                                                }
                                                for n in range(2016, 2020)
                                            ],
                                            value=2019,
                                            placeholder="Select the desired year",
                                        )
                                    ],
                                ),
                            ],
                        ),
                        dcc.Markdown(
                            children=[
                                "Source: [FiveThirtyEight](https://github.com/fivethirtyeight/uber-tlc-foil-response/tree/master/uber-trip-data)"
                            ]
                        ),
                    ],
                ),
                # Column for app graphs and plots
                html.Div(
                    className="eight columns div-for-charts bg-grey",
                    children=[
                        dcc.Graph(id="map-graph", figure=figure_mapbox),
                        html.Div(
                            className="text-padding",
                            children=[
                                "To be defined."
                            ],
                        ),
                        html.Div(
                            dcc.Graph(id="satellite_image", figure={})
                        )
                    ],
                ),
            ]
        )
    ]
)

#====================================================================
# Callbacks
#====================================================================

# location on the map
@app.callback(
    Output(component_id="map-graph", component_property="figure"),
    [Input(component_id="slct_lake", component_property="value")]
)
def mapbox_map(slct_lake):
    dff = df.loc[slct_lake, :]
    longit_center = (dff["min_longitude"]+dff["max_longitude"])/2
    latit_center = (dff["min_latitude"]+dff["max_latitude"])/2
    figure_mapbox.update_layout(
        mapbox = {
            'center': {'lon': longit_center, 'lat': latit_center},
            'style': MAP_STYLE,
            'zoom': 6},
        showlegend = True)
    
    return figure_mapbox


# satellite image for selected lake(2019) + mask
@app.callback(
    Output(component_id="satellite_image", component_property="figure"),
    [Input(component_id="slct_lake", component_property="value")]
)
def display_satellite_image(slct_lake):
    # satelite image
    download_image(data_frame=df, slct_lake=slct_lake)
    image = import_image()
    figure = go.Figure()
    figure.add_trace(go.Image(z=image))
    figure.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        xaxis={'showgrid': False, 'zeroline': False, 'visible': False},
        yaxis={'showgrid': False, 'zeroline': False, 'visible': False}
        )
    # mask
    X, Y = import_annotation(data_frame=df, slct_lake=slct_lake)
    for x, y in zip(X, Y):
        x = [i/8 for i in x]  #the original picture was 4096x4096 pixel
        y = [i/8 for i in y]  #scaling to 4096/512 = 8
        figure.add_trace(
            go.Scatter(
                x = x, y = y,
                mode='lines', fill='toself', hoverinfo='none',
                fillcolor='red', marker_color='red', opacity=.3
            )
        )
    return figure


if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port='8050', debug=True)