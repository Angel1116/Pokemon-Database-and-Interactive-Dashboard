import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd

# Load your data
df = pd.read_csv('Pokemon.csv')

# Add a new column for the image paths, converting the names to lowercase
df['image_path'] = df['name'].apply(lambda x: f'assets/images/{x.lower()}.png')

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.Div([
        dcc.Graph(id='radar-chart', style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-top': '20px'}),
        html.Div([
            html.Div([
                dcc.Dropdown(
                    id=f'pokemon-dropdown-{i}',
                    options=[{'label': pokemon, 'value': pokemon} for pokemon in df['name']],
                    placeholder=f'Select Pokémon {i+1}'
                ),
                html.Div(id=f'image-container-{i}')
            ], style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-bottom': '20px'}) for i in range(6)
        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'})
    ], style={'display': 'flex', 'justify-content': 'space-between'})
])

# Define the callback to update the images and radar chart based on the selected Pokémon
@app.callback(
    [Output(f'image-container-{i}', 'children') for i in range(6)] +
    [Output('radar-chart', 'figure')],
    [Input(f'pokemon-dropdown-{i}', 'value') for i in range(6)]
)
def update_content(*selected_pokemons):
    images = []
    radar_data = []

    for selected_pokemon in selected_pokemons:
        if selected_pokemon is not None:
            # Get the image path
            image_path = df[df['name'] == selected_pokemon]['image_path'].values[0]
            images.append(html.Img(src=image_path, style={'width': '100%', 'height': 'auto'}))

            # Get the stats for the selected Pokémon
            pokemon_stats = df[df['name'] == selected_pokemon][['HP', 'attack', 'defense', 'special_attack', 'special_defense', 'speed']].iloc[0]
            radar_data.append({
                'r': pokemon_stats.values,
                'theta': ['HP', 'Attack', 'Defense', 'Special Attack', 'Special Defense', 'Speed'],
                'name': selected_pokemon
            })
        else:
            images.append(html.Div())

    # Create the radar chart
    radar_chart = {
        'data': [
            {
                'type': 'scatterpolar',
                'r': data['r'],
                'theta': data['theta'],
                'fill': 'toself',
                'name': data['name']
            } for data in radar_data
        ],
        'layout': {
            'polar': {
                'radialaxis': {
                    'visible': True,
                    'range': [0, 100]
                }
            },
            'showlegend': True
        }
    }

    return images + [radar_chart]

if __name__ == '__main__':
    app.run_server(debug=True)