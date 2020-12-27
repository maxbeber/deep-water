import dash
from app_callbacks import callback_dropdown_year, callback_mapbox, callback_satellite_image, callback_pie_chart, callback_histogram
from app_helpers import load_models, load_dataset
from app_layout import render_layout
from dash.dependencies import Input, Output


mapbox_access_token = "pk.eyJ1Ijoia2FybHJhZHRrZSIsImEiOiJja2YyZnVvbzUwODJ6MnVxbHU0cDV4YXAxIn0.a3aiSNjy2BOO0WKg40PSsA"
app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
models = load_models()
df = load_dataset('assets/water-bodies-ui.json')
src_dash_logo = app.get_asset_url("dash-logo-new.png")
app.layout = render_layout(df, mapbox_access_token, src_dash_logo)


@app.callback(
    Output(component_id="dropdown_year", component_property="options"),
    [Input(component_id="dropdown_water_body", component_property="value")]
    )
def update_dropdown_year(dropdown_water_body):
    years = callback_dropdown_year(df, dropdown_water_body)
    return years


@app.callback(
    Output(component_id="map-graph", component_property="figure"),
    Input(component_id="dropdown_water_body", component_property="value")
)
def mapbox_map(dropdown_water_body):
    mapbox = callback_mapbox(df, mapbox_access_token, dropdown_water_body)
    return mapbox


@app.callback(
    Output(component_id="satellite_image", component_property="figure"),
    [Input(component_id="dropdown_water_body", component_property="value"),
    Input(component_id="dropdown_year", component_property="value"),
    Input(component_id="slider_opacity", component_property="value")]
)
def display_satellite_image(dropdown_water_body, dropdown_year, slider_opacity):
    figure = callback_satellite_image(df, models, dropdown_water_body, dropdown_year, slider_opacity)
    return figure


@app.callback(
    Output("histogram", "figure"),
    [Input(component_id="dropdown_water_body", component_property="value")]
)
def update_histogram(dropdown_water_body):
    histogram = callback_histogram(df, models, dropdown_water_body)
    return histogram


@app.callback(
    Output(component_id="pie_chart", component_property="figure"),
    [Input(component_id="dropdown_water_body", component_property="value"),
    Input(component_id="dropdown_year", component_property="value")]
)
def update_pie_chart(dropdown_water_body, dropdown_year):
    pie_chart = callback_pie_chart(df, models, dropdown_water_body, dropdown_year)
    return pie_chart


if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port='8050', debug=True)