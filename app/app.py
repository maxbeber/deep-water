import dash as dash
import dash_table
import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import os
#from preprocessing.waterMask import WaterMask

app = dash.Dash(__name__)



#----------------------------------------------------------------
#import data
df = pd.read_json('../datasets/waterBodies.json').\
    T.\
    explode("layers").\
    rename(columns={"layers": "year"})
df_groupby = df.\
    groupby(["country", "year"])["min_longitude"].\
    max().reset_index()


#water_mask = WaterMask('nwpu_images')

#----------------------------------------------------------------
#App layout
app.layout = html.Div([
    html.H1("1) water bodies"),

    

    dash_table.DataTable(
        id='table',
        columns=[
            {"name": i, "id": i, "selectable" : True} for i in df.columns
            ],
        data=df.to_dict('records'),
        page_action='none',
        fixed_rows={'headers': True},
        style_table={'height': '400px'},
        sort_action='native',
        row_selectable='single',
        style_cell_conditional=[    # align text columns to left. By default they are aligned to right
            {
                'if': {'column_id': c},
                'textAlign': 'left'
            } for c in ['name', 'country']
            ],
        ),
    
    dcc.Dropdown(
        id="slct_year", 
        options=[
            {"label": "2016", "value":2016},
            {"label": "2017", "value":2017},
            {"label": "2018", "value":2018}
        ],
        multi=False,
        value=2016,
        style={'width': "40%"},
    ),
    html.Br(),

    html.Div(id='output_container', children=[]),
    

    dcc.Graph(id='my_useless_graph', figure={}),

    dcc.Graph(id='slctd_min_latitude', figure={}),
    
    
    html.H1('training data'),
    dcc.Graph(id='training', figure={}),
])

#----------------------------------------------------------------
# connect plotly to graphs with dashboard
@app.callback(
    [Output(component_id="output_container", component_property='children'),
    Output(component_id="my_useless_graph", component_property='figure')],
    [Input(component_id="slct_year", component_property="value")]
)
def update_graph(option_scltd):
    #print(option_scltd)
    #print(type(option_scltd))

    container = "The year chosen by user was: {}".format(option_scltd)

    dff = df_groupby.copy()
    dff = dff[dff["year"] == option_scltd]

    fig = px.bar(
        data_frame=dff, x="country", y="min_longitude" 
    )
    return container, fig

#----------------------------------------------------------------
# extract coordinates upon clicking
@app.callback(
    Output(component_id='slctd_min_latitude', component_property='figure'),
    [
        Input(component_id='table', component_property="derived_virtual_data"),
        Input(component_id='table', component_property='derived_virtual_selected_rows')
        ]
    )
def update_coordinates(all_rows_data, slctd_row_indices):
    dff = pd.DataFrame(all_rows_data)
    if not slctd_row_indices:
        slctd_row_indices = [6]

    print(slctd_row_indices)


    longit = dff.loc[slctd_row_indices, "min_longitude"].values.tolist()
    longit = longit + dff.loc[slctd_row_indices, "max_longitude"].values.tolist()
    longit = longit + dff.loc[slctd_row_indices, "max_longitude"].values.tolist()
    longit = longit + dff.loc[slctd_row_indices, "min_longitude"].values.tolist()

    latit = dff.loc[slctd_row_indices, "min_latitude"].values.tolist()
    latit = latit + latit
    latit = latit + dff.loc[slctd_row_indices, "max_latitude"].values.tolist()
    latit = latit + dff.loc[slctd_row_indices, "max_latitude"].values.tolist()
        
    print(longit[0])

   
    fig = go.Figure(go.Scattermapbox(
        fill = "toself", lon = longit, lat = latit, 
        marker = { 'size': 1, 'color': "orange"}))
    
    fig.update_layout(
    mapbox = {
        'style': "stamen-terrain",
        'center': {'lon': longit[0], 'lat': latit[0]},
        'zoom': 8},
    showlegend = False)
    
    return fig

#----------------------------------------------------------------
# 

if __name__ == '__main__':
    app.run_server(debug=True)
