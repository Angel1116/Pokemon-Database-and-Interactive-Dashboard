from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import dash
import os

# Load the Gym.csv and Gym_Pokemon.csv files into DataFrames
gym_df = pd.read_csv('Gym.csv')
gym_pokemon_df = pd.read_csv('Gym_Pokemon.csv')

# Merge the DataFrames on the gym_id column
merged_df = pd.merge(gym_df, gym_pokemon_df, on='gym_id')

# Create a new column for the gym leader image paths, ensuring the leader names are in lowercase
merged_df['leader_image_path'] = merged_df['leader'].apply(lambda x: f"assets/gen1_leaders/{x.lower()}.png")

# Create a new column for the Pokémon image paths, ensuring the Pokémon names are in lowercase
merged_df['pokemon_image_path'] = merged_df['pokemon'].apply(lambda x: f"assets/images/{x.lower()}.png")

# Load the Pokemon.csv file into a DataFrame
df = pd.read_csv('Pokemon.csv')

# Add a new column for the image paths, converting the names to lowercase
df['image_path'] = df['name'].apply(lambda x: f'assets/images/{x.lower()}.png')

# Load player images
player_images = [f"assets/players/{img}" for img in os.listdir('assets/players') if img.endswith('.png')]

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    # Top row with Gym Leader and their Pokémon
    html.Div([
        # Gym Leader Dropdown and Info
        html.Div([
            dcc.Dropdown(
                id='gym-leader-dropdown',
                options=[{'label': leader, 'value': leader} for leader in merged_df['leader'].unique()],
                placeholder="Select a Gym Leader",
                style={'width': '200px', 'display': 'inline-block'}
            ),
            html.Div(id='leader-info', style={'margin-top': '20px'})
        ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
        
        # Gym Leader's Pokémon Images
        html.Div(id='pokemon-images', style={
            'width': '70%', 
            'display': 'grid', 
            'grid-template-columns': 'repeat(3, 1fr)', 
            'grid-gap': '10px',
            'justify-content': 'center'
        })
    ], style={'width': '100%', 'display': 'flex'}),

    # Second row with Player and Pokémon Dropdowns
    html.Div([
        # Player Character Selection
        html.Div([
            dcc.Dropdown(
                id='player-dropdown',
                options=[{'label': img.split('/')[-1].split('.')[0], 'value': img} for img in player_images],
                placeholder="Select Your Character",
                style={'width': '200px', 'display': 'inline-block'}
            ),
            html.Div(id='player-image', style={'margin-top': '20px'})
        ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
        
        # Pokémon Dropdowns and Images
        html.Div([
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id=f'pokemon-dropdown-{i}',
                        options=[{'label': pokemon, 'value': pokemon} for pokemon in df['name']],
                        placeholder=f'Select Pokémon {i+1}',
                        style={'width': '100%', 'margin-bottom': '10px'}
                    ),
                    html.Div(id=f'image-container-{i}', style={
                        'width': '100%', 
                        'height': '150px', 
                        'display': 'flex', 
                        'justify-content': 'center', 
                        'align-items': 'center'
                    })
                ], style={
                    'width': '33.33%', 
                    'padding': '5px', 
                    'box-sizing': 'border-box'
                }) for i in range(6)
            ], style={
                'width': '100%', 
                'display': 'flex', 
                'flex-wrap': 'wrap',
                'justify-content': 'center'
            })
        ], style={'width': '70%', 'display': 'inline-block', 'vertical-align': 'top'})
    ], style={'width': '100%', 'display': 'flex', 'margin-top': '20px'}),

    # Radar Chart
    html.Div([
        dcc.Graph(id='radar-chart', style={'width': '100%', 'height': '400px'})
    ], style={'width': '100%', 'margin-top': '20px'})
])

# Define the callback to update the leader info and Pokémon images
@app.callback(
    [Output('leader-info', 'children'), Output('pokemon-images', 'children')],
    [Input('gym-leader-dropdown', 'value')]
)
def update_leader_info(selected_leader):
    if selected_leader is None:
        return "", ""

    # Filter the DataFrame for the selected leader
    leader_df = merged_df[merged_df['leader'] == selected_leader]

    # Get the leader image path
    leader_image_path = leader_df['leader_image_path'].iloc[0]

    # Create the leader info div
    leader_info = html.Div([
        html.Img(src=leader_image_path, style={'height': '200px'}),
        html.H3(selected_leader)
    ])

    # Create the Pokémon images div
    pokemon_images = [
        html.Div([
            html.Img(src=row['pokemon_image_path'], style={'max-width': '100%', 'max-height': '200px'})
        ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}) for _, row in leader_df.iterrows()
    ]

    return leader_info, pokemon_images

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
            images.append(html.Img(src=image_path, style={'max-width': '100%', 'max-height': '150px'}))

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

# Define the callback to update the player image based on the selected character
@app.callback(
    Output('player-image', 'children'),
    [Input('player-dropdown', 'value')]
)
def update_player_image(selected_player):
    if selected_player is None:
        return ""

    return html.Img(src=selected_player, style={'height': '200px'})

if __name__ == '__main__':
    app.run_server(debug=True)