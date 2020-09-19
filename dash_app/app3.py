import dash as dash
import dash_table
import numpy as np
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
import utils.dash_reusable_components as drc
from dash.dependencies import Input, Output
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from app_helpers import download_image, import_image, lat_long,\
    import_annotation, load_model, model_prediction, blkwhte_rgb, slct_image,\
    calculate_water, get_water_land_per_year, get_sqkm, get_geom
import matplotlib.pyplot as plt

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)

#====================================================================
# prerequisites
#--------------------------------------------------------------------
# import dataset
df = pd.read_json('waterBodiesDash.json').T
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
        height=300)
    mapbox.update_layout(mapbox_style=MAP_STYLE, mapbox_accesstoken=mapbox_access_token)
    mapbox.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return mapbox

# Build map box
figure_mapbox = build_mapbox()

#====================================================================
# app laypout
#====================================================================
model_prediction = [
    html.P("Model Prediction"),
    html.Div(
        dcc.Graph(id="satellite_image", figure={})
        ),
    html.P("Surface Area (square kilometer)"),
    html.Div(
        dcc.Graph(id="pie_chart", figure={})
        ) 
        ]

geo_location = [
    html.P("Geo-location"),
    html.Div(
        dcc.Graph(id="map-graph", figure=figure_mapbox)),
            html.P(""),
            html.P("Surface Area (%)"),
            html.Div(
            dcc.Graph(id="histogram")
        )   
    ]

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
                                )
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
                    className="one-third column div-for-charts bg-grey",
                    children=model_prediction
                ),
                html.Div(
                    className="one-third column div-for-charts bg-grey",
                    children=geo_location
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
    Input(component_id="dropdown_year", component_property="value"),
    Input(component_id="slider_opacity", component_property="value")]
)
def display_satellite_image(dropdown_water_body, dropdown_year, slider_opacity):
    figure = go.Figure()
    figure.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        xaxis={'showgrid': False, 'zeroline': False, 'visible': False},
        yaxis={'showgrid': False, 'zeroline': False, 'visible': False},
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        height=300,
        showlegend=False,
    )
    if not dropdown_water_body:
        return figure
    # satelite image
    lake = slct_image(
        data_frame=df,
        slct_lake=dropdown_water_body,
        slct_year=dropdown_year
        )

    image = import_image(lake)
    figure.add_trace(go.Image(z=image))
    #mask
    model = load_model()
    mask = model.predict(np.expand_dims(image, axis=0))
    mask = blkwhte_rgb(mask)

    figure.add_trace(
        go.Image(z=mask, opacity=slider_opacity)
        )
    
    return figure


@app.callback(
    Output("histogram", "figure"),
    [Input(component_id="dropdown_water_body", component_property="value")]
)
def update_histogram(dropdown_water_body):
    if not dropdown_water_body:
        figure = go.Figure()
        figure.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            xaxis={'showgrid': False, 'zeroline': False, 'visible': False},
            yaxis={'showgrid': False, 'zeroline': False, 'visible': False},
            plot_bgcolor="#282b38",
            paper_bgcolor="#282b38")
        return figure
    
    dff = df.loc[dropdown_water_body, :]
    years = dff["layers"]

    model = load_model()
    prediction = []
    prediction_dic = dict()
    for i in years:
        lake = slct_image(
            data_frame=df,
            slct_lake=dropdown_water_body,
            slct_year=i
            )
        image = import_image(lake)
        mask = model.predict(np.expand_dims(image, axis=0))
        water_percentage = calculate_water(mask) * 100
        prediction_dic[str(i)] = water_percentage
        prediction.append(water_percentage)


    [xVal, yVal, _] = [years, prediction, ['#000']]
    layout = go.Layout(
        bargap=0.01,
        bargroupgap=0,
        barmode="group",
        margin=go.layout.Margin(l=10, r=0, t=0, b=30),
        showlegend=False,
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        dragmode="select",
        font=dict(color="white"),
        height=150,
        xaxis=dict(
            range=[2015, 2020],
            showgrid=False,
            fixedrange=True
        ),
        yaxis=dict(
            range=[0, max(yVal) + max(yVal) / 4],
            showticklabels=False,
            showgrid=False,
            fixedrange=True,
            rangemode="nonnegative",
            zeroline=False,
        ),
        annotations=[
            dict(
                x=xi,
                y=yi,
                text=str(np.round(yi, 2)),
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font=dict(color="white"),
            )
            for xi, yi in zip(xVal, yVal)
        ],
    )

    histogram = go.Figure(
        data=[
            #go.Bar(x=xVal, y=yVal, marker=dict(color='blue'), hoverinfo="x"),
            go.Scatter(
                opacity=1,
                x=xVal,
                y=yVal,
                hoverinfo="none",
                mode="lines+markers",
                marker=dict(color="rgb(66, 134, 244, 0)", size=40),
                visible=True,
            ),
        ],
        layout=layout,
    )

    return histogram


@app.callback(
    Output(component_id="pie_chart", component_property="figure"),
    [Input(component_id="dropdown_water_body", component_property="value"),
    Input(component_id="dropdown_year", component_property="value")]
)
def update_pie_chart(dropdown_water_body, dropdown_year):
    if not dropdown_water_body:
        figure = go.Figure()
        figure.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            xaxis={'showgrid': False, 'zeroline': False, 'visible': False},
            yaxis={'showgrid': False, 'zeroline': False, 'visible': False},
            plot_bgcolor="#282b38",
            paper_bgcolor="#282b38")
        return figure

    dff = df.loc[dropdown_water_body, :]

    lake = slct_image(
        data_frame=df,
        slct_lake=dropdown_water_body,
        slct_year=dropdown_year
        )

    image = import_image(lake)
    #figure.add_trace(go.Image(z=image))
    #mask
    model = load_model()
    mask = model.predict(np.expand_dims(image, axis=0))
    bounding_box = get_geom(dff)
    image_sqkm = get_sqkm(bounding_box)
    water_percentage = calculate_water(mask)
    water_sqkm, land_sqkm = get_water_land_per_year(
        fraction = water_percentage, area=image_sqkm)
    water_sqkm, land_sqkm = round(water_sqkm, 2), round(land_sqkm, 2)
    labels=["water", "land"]
    values=[water_sqkm, land_sqkm]
    
    piechart = go.Figure(data=[go.Pie(labels=labels, values=values)])
    piechart.update_traces(
        hoverinfo='none',
        textinfo='value',#, textfont_size=20#,
        marker=dict(colors=["rgb(66, 134, 244, 0)", "rgb(150, 75, 0)"])
        )
    piechart.update_layout(plot_bgcolor="#282b38",
        paper_bgcolor="#282b38")

    return piechart



if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port='8050', debug=True)