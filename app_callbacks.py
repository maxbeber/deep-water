import numpy as np
import plotly.graph_objects as go
from app_helpers import calculate_water, ensemble_predict, get_bounding_box, get_bounding_box_area, get_image_path, get_water_land_per_year, load_image
from app_layout import get_mapbox


def callback_dropdown_year(df, dropdown_water_body):
    if not dropdown_water_body:
        return [{"label": str(n), "value": str(n)} for n in range(2016, 2020)]
    layers = df.loc[dropdown_water_body, "layers"]
    years = [{"label": str(n), "value": str(n)} for n in layers]

    return years


def callback_histogram(df, models, dropdown_water_body):
    if not dropdown_water_body:
        histogram = _get_histogram_default()
        return histogram
    years = df.loc[dropdown_water_body, "layers"]
    prediction = []
    prediction_dic = dict()
    for i in years:
        lake = get_image_path(df, dropdown_water_body, i)
        image = load_image(lake)
        mask = ensemble_predict(models, image)
        water_percentage = calculate_water(mask) * 100
        prediction_dic[str(i)] = water_percentage
        prediction.append(water_percentage)
    [X, Y, _] = [years, prediction, ['#000']]
    histogram = _get_histogram(X, Y)

    return histogram


def callback_mapbox(df, mapbox_access_token, dropdown_water_body):
    mapbox = get_mapbox(df, mapbox_access_token)
    if not dropdown_water_body:
        return mapbox
    dff = df.loc[dropdown_water_body, ["min_longitude", "max_longitude", "min_latitude", "max_latitude"]]
    min_longitude = dff["min_longitude"]
    max_longitude = dff["max_longitude"]
    min_latitude = dff["min_latitude"]
    max_latitude = dff["max_latitude"]
    longit_center = (min_longitude + max_longitude) / 2.0
    latit_center = (min_latitude + max_latitude) / 2.0
    mapbox.update_layout(
        mapbox = {
            'center': {'lon': longit_center, 'lat': latit_center},
            'style': 'outdoors',
            'zoom': 6},
        showlegend = True)

    return mapbox


def callback_pie_chart(df, models, dropdown_water_body, dropdown_year):
    if not dropdown_water_body:
        pie_chart = _get_pie_chart_default()
        return pie_chart
    dff = df.loc[dropdown_water_body, :]
    # get predictions from the model
    lake = get_image_path(df, dropdown_water_body, dropdown_year)
    image = load_image(lake)
    mask = ensemble_predict(models, image)
    # receiving the area for the whole image
    bounding_box = get_bounding_box(dff)
    image_sqkm = get_bounding_box_area(bounding_box)
    # calculating the area for water and land
    water_percentage = calculate_water(mask)
    water_sqkm, land_sqkm = get_water_land_per_year(fraction=water_percentage, area=image_sqkm)
    water_sqkm, land_sqkm = round(water_sqkm, 2), round(land_sqkm, 2)
    #plot
    values=[water_sqkm, land_sqkm]
    pie_chart = _get_pie_chart(values)

    return pie_chart


def callback_satellite_image(df, models, dropdown_water_body, dropdown_year, slider_opacity):
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
    lake = get_image_path(df, dropdown_water_body, dropdown_year)
    image = load_image(lake)
    mask = ensemble_predict(models, image)
    _apply_mask_contour(figure, image, mask, slider_opacity)
    
    return figure


def _apply_mask_contour(figure, image, mask, slider_opacity):
    colorscale = [[0, 'gold'], [0.5, 'gold'], [1, 'gold']]
    image = 255 * image
    figure.add_trace(
        go.Image(z=image)
    )
    figure.update_layout(
        margin={"r": 60, "t": 0, "l": 0, "b": 0}
    )
    figure.add_trace(
        go.Contour(
            z=mask,
            contours_coloring='lines',
            line_width=3,
            opacity=slider_opacity,
            showlegend=False,
            showscale=False,
            colorscale=colorscale,
            colorbar=dict(showticklabels=False))
        )
    
    return figure


def _get_histogram_default():
    figure = go.Figure()
    figure.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        xaxis={'showgrid': False, 'zeroline': False, 'visible': False},
        yaxis={'showgrid': False, 'zeroline': False, 'visible': False},
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38")

    return figure


def _get_histogram(X, Y):
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
            range=[2015.5, 2019.5],
            showgrid=False,
            fixedrange=False
        ),
        yaxis=dict(
            range=[0, max(Y) + max(Y) / 4],
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
            for xi, yi in zip(X, Y)
        ],
    )
    histogram = go.Figure(
        data=[
            go.Scatter(
                opacity=1,
                x=X,
                y=Y,
                hoverinfo="none",
                mode="lines+markers",
                marker=dict(color="rgb(66, 134, 244, 0)", size=40),
                visible=True,
            ),
        ],
        layout=layout,
    )

    return histogram


def _get_pie_chart_default():
    figure = go.Figure()
    figure.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        xaxis={'showgrid': False, 'zeroline': False, 'visible': False},
        yaxis={'showgrid': False, 'zeroline': False, 'visible': False},
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38")

    return figure


def _get_pie_chart(values):
    labels=["water", "land"]
    figure = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.3,
                showlegend=False)
            ]
        )
    figure.update_traces(
        hoverinfo='label',
        textinfo='value',
        marker=dict(colors=["rgb(66, 134, 244, 0)", "rgb(150, 75, 0)"])
    )
    figure.update_layout(
        margin={"r":100, "t":0, "l":0, "b":0},
        height=150,
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38"
    )

    return figure