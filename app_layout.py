import dash_html_components as html
import dash_core_components as dcc
import plotly.express as px
import plotly.graph_objects as go
import tools.dash_reusable_components as drc


MAP_STYLE= 'outdoors'


def get_mapbox(df, mapbox_access_token):
    mapbox = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        center=go.layout.mapbox.Center(lat=52.52,lon=13.4050),
        hover_name="name",
        color_discrete_sequence=["fuchsia"],
        zoom=8,
        height=300)
    mapbox.update_layout(
        margin={"r":0, "t":0, "l":0, "b":0},
        mapbox_style=MAP_STYLE,
        mapbox_accesstoken=mapbox_access_token
    )
    return mapbox


def render_layout(df, mapbox_access_token, src_logo):
    layout = _build_layout(df, mapbox_access_token, src_logo)
    return layout


def _build_geo_location_section(df, mapbox_access_token):
    ctrl_mapbox = get_mapbox(df, mapbox_access_token)
    layout = [
        html.H3("Geolocation"),
        html.Div(
            dcc.Graph(id="map-graph", figure=ctrl_mapbox)
        ),
        html.H3("Water Surface (%)"),
        html.Div(
            dcc.Graph(id="histogram")
        )   
    ]
    section = html.Div(
        className="one-third column div-for-charts bg-grey",
        children=layout
    ) 

    return section


def _build_layout(df, mapbox_access_token, src_logo):
    menu = _build_menu(df, src_logo)
    model_geo_location_section = _build_geo_location_section(df, mapbox_access_token)
    model_prediction_section = _build_model_prediction_section()
    section = html.Div(
        children=[
            html.Div(
                className="row",
                children=[
                    menu,
                    model_prediction_section,
                    model_geo_location_section
                ]
            )
        ]   
    )

    return section


def _build_menu(df, src_logo):
    copyright_section = _build_menu_copyright()
    dropdown_opacity = _build_menu_dropdown_opacity()
    dropdown_water_body = _build_menu_dropdown_water_body(df)
    dropdown_year = _build_menu_dropdown_year()
    section = html.Div(
        className="four columns div-user-controls",
        children=[
            html.Img(className="logo", src=src_logo),
            html.H2("DEEP WATER"),
            html.P("""Select a water body and the desired year."""),
            html.Div(
                className="row",
                children=[
                    dropdown_water_body,
                    dropdown_year,
                    dropdown_opacity
                ]
            ),
            copyright_section
        ]
    )

    return section


def _build_menu_copyright():
    section = dcc.Markdown(
        children=[
            "Design by: [FiveThirtyEight](https://github.com/fivethirtyeight/uber-tlc-foil-response/tree/master/uber-trip-data)"
        ]
    )
    return section


def _build_menu_dropdown_opacity():
    slider_marks = {str(i): str(i) for i in [0.2, 0.4, 0.6, 0.8, 1.0]}
    section = html.Div(
        className="div-for-dropdown",
        children=[
            drc.NamedSlider(
                name="Mask Opacity",
                id="slider_opacity",
                min=0,
                max=1,
                step=0.1,
                marks=slider_marks,
                value=0.8,
            )
        ]
    )

    return section


def _build_menu_dropdown_water_body(df):
    options = [{"label": name.replace('_', ' ').capitalize() + ', ' + country.replace('_', ' ').capitalize(), "value":i} for name, country, i in zip(df["name"], df["country"], df.index)]
    section = html.Div(
        className="div-for-dropdown",
        children=[
            dcc.Dropdown(
                id="dropdown_water_body",
                options=options,
                placeholder="Select a water body"
            )
        ]
    )

    return section


def _build_menu_dropdown_year():
    section = html.Div(
        className="div-for-dropdown",
        children=[
            dcc.Dropdown(
                id="dropdown_year",
                options=[],
                value=2019,
                placeholder="Select the desired year",
            )
        ]
    )
    return section


def _build_model_prediction_section():
    layout = [
        html.H3("Model Prediction"),
        html.Div(
            dcc.Graph(id="satellite_image", figure={})
        ),
        html.H3("Surface Area (square kilometer)"),
        html.Div(
            dcc.Graph(id="pie_chart", figure={})
        ) 
    ]
    section = html.Div(
        className="one-third column div-for-charts bg-grey",
        children=layout
    )

    return section