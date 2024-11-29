import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Load the data from the CSV file
df = pd.read_csv('output.csv')

# Assuming you have a column 'pokemon' in your DataFrame that matches the image filenames
# Add a new column for the image paths
df['image_path'] = df['pokemon'].apply(lambda x: f'assets/images/{x}.png')

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    dcc.Dropdown(
        id='pokemon-dropdown',
        options=[{'label': pokemon, 'value': pokemon} for pokemon in df['pokemon']],
        placeholder='Select a Pokémon'
    ),
    html.Div(id='image-container'),
    dcc.Graph(id='radar-chart')
])

# Define the callback to update the image and radar chart based on the selected Pokémon
@app.callback(
    [Output('image-container', 'children'),
     Output('radar-chart', 'figure')],
    [Input('pokemon-dropdown', 'value')]
)
def update_content(selected_pokemon):
    if selected_pokemon is None:
        return html.Div(), {}
    
    # Get the image path
    image_path = df[df['pokemon'] == selected_pokemon]['image_path'].values[0]
    
    # Get the stats for the selected Pokémon
    pokemon_stats = df[df['pokemon'] == selected_pokemon][['HP', 'attack', 'defense', 'special_attack', 'special_defense', 'speed']].iloc[0]
    
    # Create the radar chart
    fig = px.line_polar(
        r=pokemon_stats.values,
        theta=pokemon_stats.index,
        line_close=True,
        title=f'{selected_pokemon} Stats'
    )
    fig.update_traces(fill='toself')
    
    return html.Img(src=image_path, style={'width': '300px', 'height': '300px'}), fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)