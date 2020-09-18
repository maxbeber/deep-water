import dash as dash
import dash_table
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
import utils.dash_reusable_components as drc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import os
from app_helpers import download_image, import_image, lat_long, import_annotation, load_model, model_prediction, blkwhte_rgb
import matplotlib.pyplot as plt

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

# Constants
MAP_STYLE= 'outdoors'

def build_mapbox():
    mapbox = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon", 
        hover_name="name",
        color_discrete_sequence=["fuchsia"],
        zoom=3,
        height=200)
    mapbox.update_layout(mapbox_style=MAP_STYLE, mapbox_accesstoken=mapbox_access_token)
    mapbox.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return mapbox

# Build map box
figure_mapbox = build_mapbox()

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
                                            id="dropdown_water_body",
                                            options=[
                                                {"label": name.capitalize() + ', ' + country.replace('_', ' ').capitalize(), "value":i} for name, country, i in zip(df["name"], df["country"], df.index)
                                            ],
                                            placeholder="Select a water body",
                                        )
                                    ],
                                ),
                                html.Div(
                                    className="div-for-dropdown",
                                    children=[
                                        # Dropdown to select desired year
                                        dcc.Dropdown(
                                            id="dropdown_year",
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
                                html.Div(
                                    className="div-for-dropdown",
                                    children=[
                                        # Slider to control the mask opacity
                                        drc.NamedSlider(
                                            name="Mask Opacity",
                                            id="slider_opacity",
                                            min=0,
                                            max=1,
                                            step=0.1,
                                            marks={
                                                str(i): str(i)
                                                for i in [0.2, 0.4, 0.6, 0.8, 1.0]
                                            },
                                            value=0.2,
                                        )
                                    ],
                                ),
                            ],
                        ),
                        dcc.Markdown(
                            children=[
                                "Design by: [FiveThirtyEight](https://github.com/fivethirtyeight/uber-tlc-foil-response/tree/master/uber-trip-data)"
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
    Input(component_id="dropdown_water_body", component_property="value")
)
def mapbox_map(dropdown_water_body):
    if not dropdown_water_body:
        return figure_mapbox
    dff = df.loc[dropdown_water_body, :]
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
    [Input(component_id="dropdown_water_body", component_property="value"),
    Input(component_id="slider_opacity", component_property="value")]
)
def display_satellite_image(dropdown_water_body, slider_opacity):
    figure = go.Figure()
    figure.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        xaxis={'showgrid': False, 'zeroline': False, 'visible': False},
        yaxis={'showgrid': False, 'zeroline': False, 'visible': False}
    )
    if not dropdown_water_body:
        return figure
    # satelite image
    download_image(data_frame=df, slct_lake=dropdown_water_body)
    image = import_image()
    figure.add_trace(go.Image(z=image))
    #mask
    model = load_model()
    image_path = 'sample_image.jpg'
    mask = model_prediction(X=image_path, model=model)

    mask = blkwhte_rgb(mask)

    
    figure.add_trace(
        go.Image(z=mask, opacity=slider_opacity)
        )
    
    return figure


if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port='8050', debug=True)