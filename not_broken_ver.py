from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import dash
import os
import plotly.graph_objects as go
import sqlite3
import matplotlib.pyplot as plt
import numpy as np

# Load the Gym.csv and Gym_Pokemon.csv files into DataFrames
gym_df = pd.read_csv('Gym.csv')
gym_pokemon_df = pd.read_csv('Gym_Pokemon.csv')
# Load the Pokemon.csv file into a DataFrame
df = pd.read_csv('Pokemon.csv')

# Merge the DataFrames on the gym_id column
merged_df = pd.merge(gym_df, gym_pokemon_df, on='gym_id')
merged_df = pd.merge(merged_df, df, left_on='pokemon', right_on='name')

# Create a new column for the gym leader image paths, ensuring the leader names are in lowercase
merged_df['leader_image_path'] = merged_df.apply(lambda row: f"assets/gen1_leaders/{row['location'].capitalize()}_{row['leader'].lower()}.png", axis=1)

# Create a new column for the Pokémon image paths, ensuring the Pokémon names are in lowercase
merged_df['pokemon_image_path'] = merged_df['pokemon'].apply(lambda x: f"assets/images/{x.lower()}.png")

# Add a new column for the image paths, converting the names to lowercase
df['image_path'] = df['name'].apply(lambda x: f'assets/images/{x.lower()}.png')

# Load player images
player_images = [f"assets/players/{img}" for img in os.listdir('assets/players') if img.endswith('.png')]

# bar chart
result_df = merged_df[['gym_id', 'generation', 'pokemon', 'leader', 'type1', 'type2']]

#connect all tables
conn = sqlite3.connect("db/pokemonStats_1127.db")
with open("all_tables.sql", "r", encoding="utf-8") as sql_file:
    sql_query = sql_file.read()
all_df = pd.read_sql_query(sql_query, conn)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    # Title
    html.Div([
        html.Div([
            html.H2("Pokémon Gym Leader Battle Simulator", style={'margin': 0}),
            html.H3("Description: ", style={'margin': 0, 'font-weight': 'normal'})
        ], style={
            'display': 'inline-block',
            'vertical-align': 'top',
            'width': '70%',
            'box-sizing': 'border-box'
        }),

        # Instructions Section
        html.Div(
            "Suggestion, Recommendation, and Instruction: This is a Pokémon Gym Leader Battle Simulator. ",
            style={
                'display': 'inline-block',
                'text-align': 'left',
                'vertical-align': 'top',
                'width': '30%',
                'font-size': '20px',
                'box-sizing': 'border-box'
            }
        )
        ], style={
        'display': 'flex',
        'width': '100%',
        'justify-content': 'space-between',
        'align-items': 'flex-start',  # Aligns items at the top
        'box-sizing': 'border-box'
        }),
    # Top row with Gym Leader and their Pokémon
    html.Div([
        # Gym Leader Dropdown and Info
        html.Div([
            dcc.Dropdown(
                id='gym-leader-dropdown',
                options=[{'label': leader, 'value': leader} for leader in merged_df['leader'].unique()],
                placeholder="Select a Gym Leader",
                style={'width': '200px', 'display': 'inline-block'}  # Increased width to match
            ),
            html.Div(id='leader-info', style={'margin-top': '20px'})
        ], style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top'}),

        # Gym Leader's Pokémon Images
        html.Div(id='pokemon-images', style={
            'width': '60%', 
            'display': 'grid', 
            'grid-template-columns': 'repeat(3, 1fr)', 
            'grid-gap': '2px',
            'justify-content': 'center'
        }),

        html.Div([
            dcc.Dropdown(
                id='player-dropdown',
                options=[{'label': img.split('/')[-1].split('.')[0], 'value': img} for img in player_images],
                placeholder="Select Your Character",
                style={'width': '200px', 'display': 'inline-block'}
            ),
            html.Div(id='player-image', style={'margin-top': '20px'})
        ], style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top'}),
        
        # Pokémon Dropdowns and Images
        html.Div([
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id=f'pokemon-dropdown-{i}',
                        options=[{'label': pokemon, 'value': pokemon} for pokemon in df['name']],
                        placeholder=f'Select Pokémon {i+1}',
                        style={'width': '100%', 'margin-top': '20px'}
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
                    'box-sizing': 'border-box',
                    'margin-top': '10px'
                }) for i in range(6)
            ], style={
                'width': '100%', 
                'display': 'flex', 
                'flex-wrap': 'wrap',
                'justify-content': 'center',
                'align-items': 'flex-start'
            })
        ], style={'width': '70%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-top': '0px'}),
    ], style={'width': '100%', 'display': 'flex'}),

    html.Div([
        # Bar Chart 1
        html.Div([
            dcc.Graph(id="bar-chart-1", style={'width': '100%'})
        ], style={'width': '50%', 'padding': '10px'}),  # Half the page width

        html.Div("Instructions for Chart 1: Describe what this chart represents and how to interpret it.", 
                 style={'margin-top': '10px', 'font-size': '14px', 'color': 'gray'}),

        # Bar Chart 2
        html.Div([
            dcc.Graph(id="bar-chart-2")
        ], style={'width': '50%', 'padding': '10px'}),  # Half the page width

        html.Div("Instructions for Chart 2: Describe what this chart represents and how to interpret it.", 
                 style={'margin-top': '10px', 'font-size': '14px', 'color': 'gray'})
    ], style={
        'display': 'flex', 
        'width': '100%', 
        'justify-content': 'space-between',  # Adds spacing between the charts
        'align-items': 'center'  # Aligns charts vertically in case of differing heights
    }),
    html.Div([
        # Radar Chart
        html.Div([
            dcc.Graph(id='radar-chart', style={'width': '100%', 'height': '400px'})
        ], style={'width': '50%', 'margin-top': '20px'}),

        # Distribution Line
        html.Div([
            dcc.Graph(id='distribution-line', style={'width': '100%', 'height': '400px'})
        ], style={'width': '100%', 'margin-top': '20px'}),

        # Heatmap
        html.Div([
            dcc.Graph(id='heatmap', style={'width': '100%', 'height': '400px'})
        ], style={'width': '100%', 'margin-top': '20px'})


    ],style={
        'display': 'flex', 
        'width': '100%', 
        'justify-content': 'space-between',  # Adds spacing between the charts
        'align-items': 'center'  # Aligns charts vertically in case of differing heights
    })
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
    team_stats = {
        'HP': 0,
        'Attack': 0,
        'Defense': 0,
        'Special Attack': 0,
        'Special Defense': 0,
        'Speed': 0
    }

    for selected_pokemon in selected_pokemons:
        if selected_pokemon is not None:
            # Get the image path
            image_path = df[df['name'] == selected_pokemon]['image_path'].values[0]
            images.append(html.Img(src=image_path, style={'max-width': '100%', 'max-height': '150px'}))

            # Get the stats for the selected Pokémon
            pokemon_stats = df[df['name'] == selected_pokemon].iloc[0]
            individual_stats = {
                'HP': pokemon_stats['HP'],
                'Attack': pokemon_stats['attack'],
                'Defense': pokemon_stats['defense'],
                'Special Attack': pokemon_stats['special_attack'],
                'Special Defense': pokemon_stats['special_defense'],
                'Speed': pokemon_stats['speed']
            }

            # Add individual pokemon stats to team stats
            for stat_name in team_stats.keys():
                team_stats[stat_name] += individual_stats[stat_name]
        else:
            images.append(html.Div())

    # Pad with empty divs if fewer than 6 Pokémon are selected
    while len(images) < 6:
        images.append(html.Div())

    # Create the radar chart with only team stats
    radar_chart = {
        'data': [
            {
                'type': 'scatterpolar',
                'r': list(team_stats.values()),
                'theta': list(team_stats.keys()),
                'fill': 'toself',
                'name': 'Team Total',
                'marker': {'color': 'red'},
                'line': {'color': 'red'}
            }
        ],
        'layout': {
            'polar': {
                'radialaxis': {
                    'visible': True,
                    'range': [0, max(team_stats.values()) + 50]  # Adjust the range as needed
                }
            },
            'showlegend': True,
            'title': 'Team Total Stats'
        }
    }

    return images + [radar_chart]

#Yi-Syuan-----------------------------------------------------------------------------------------------
# Define the callback to update the distribution line based on the selected Pokémon
@app.callback(
    [Output('distribution-line', 'figure')],
    [Input(f'pokemon-dropdown-{i}', 'value') for i in range(6)]
)
def update_content(*selected_pokemons):
    images = []

    dist_data = {
        'HP': [],
        'attack': [],
        'defense': [],
        'special_attack': [],
        'special_defense': [],
        'speed': [],
    }
    dist_df = pd.DataFrame(dist_data, index=[])
    color_list = ['#F4A261','#FF6B6B', '#F7DC6F', '#6ED3CF', '#40A8C4', '#457B9D']
    colors = {}

    for idx, selected_pokemon in enumerate(selected_pokemons):
        if selected_pokemon is not None:
            image_path = df[df['name'] == selected_pokemon]['image_path'].values[0]
            images.append(html.Img(src=image_path, style={'max-width': '100%', 'max-height': '10px'}))

            pokemon_stats = df[df['name'] == selected_pokemon][['HP', 'attack', 'defense', 'special_attack', 'special_defense', 'speed']].iloc[0]
            value = [pokemon_stats[stat] for stat in pokemon_stats.index]
            dist_df.loc[selected_pokemon] = value
            color = color_list[idx % len(color_list)]
            colors[selected_pokemon] = color

    stats = df[['HP', 'attack', 'defense', 'special_attack', 'special_defense', 'speed']].agg(['min', 'max', 'mean']).transpose()
    fig = go.Figure()
    used_x_positions = {col: [] for col in dist_df.columns}

    for col in dist_df.columns:
        max_val = stats.loc[col, 'max']
        min_val = stats.loc[col, 'min']
        mean_val = stats.loc[col, 'mean']

                
        fig.add_trace(go.Scatter(
            x=[min_val], 
            y=[col], 
            mode='markers+text',
            text='Min', 
            textposition='bottom center',
            hovertext=[f'Min: {round(min_val, 2)}'],
            marker=dict(color='black', size=1, symbol='circle'),
            hoverinfo='text',
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[max_val], 
            y=[col], 
            mode='markers+text',
            text='Max', 
            textposition='bottom center',
            hovertext=[f'Max: {round(max_val, 2)}'],
            marker=dict(color='black', size=1, symbol='circle'),
            hoverinfo='text',
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[col, col],
            mode='lines',
            line=dict(color='#D3D3D3', width=6,),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[mean_val],
            y=[col],
            mode='markers+text',
            text='Average',
            textposition='bottom center',
            hovertext=[f'Average: {round(mean_val, 0)}'],
            hoverinfo='text',
            marker=dict(color='black', size=10, symbol='circle'),
            showlegend=False
        ))

    for col in dist_df.columns:
        for index, value in zip(dist_df.index, dist_df[col]):
            offset = 0
            while value + offset in used_x_positions[col]:
                offset += 0.5
            
            final_x = value + offset
            used_x_positions[col].append(final_x)
            if dist_df.columns[0] == col:
                fig.add_trace(go.Scatter(
                x=[final_x], 
                y=[col],  
                mode='markers+text',
                marker=dict(color=colors[index], size=10, symbol='circle'),  
                hovertext=[f'{index}: {value}'], 
                hoverinfo='text',
                showlegend=True,  
                name=index
            ))
            else:
                fig.add_trace(go.Scatter(
                    x=[final_x], 
                    y=[col],  
                    mode='markers+text',
                    marker=dict(color=colors[index], size=10, symbol='circle'),
                    hovertext=[f'{index}: {value}'],
                    hoverinfo='text',
                    showlegend=False
                ))

    fig.update_layout(
        xaxis=dict(zeroline=False, showgrid=True),
        yaxis=dict(showticklabels=True, tickvals=list(dist_df.columns), ticktext=dist_df.columns),
        height=500,
        width=800,
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(title="Label", x=1, y=1, traceorder="normal", bgcolor="rgba(255, 255, 255, 0)"),
        font=dict(family="Georgia", size=12, color="black")
    )

    return [fig]



# Define the callback to update the heatmap
@app.callback(
    [Output('heatmap', 'figure')],
    [Input('gym-leader-dropdown', 'value')] + 
    [Input(f'pokemon-dropdown-{i}', 'value') for i in range(6)]
)

def update_content(selected_leader, *selected_pokemons):
    if selected_leader is None or all(pokemon is None for pokemon in selected_pokemons):
        return [{
            "data": [],
            "layout": {
                "title": "No data available",
                "xaxis": {"title": "X-axis"},
                "yaxis": {"title": "Y-axis"}
            }
        }]

    else:
        leader_df = all_df[all_df['leader'] == selected_leader][['type1']]
        gym_leader_team = leader_df.iloc[:, 0].tolist()
        print(gym_leader_team)

        user_team = []
        for selected_pokemon in selected_pokemons:
            row = df[df['name'] == selected_pokemon]
            if not row.empty:
                type1_value = row['type1'].iloc[0]
                user_team.append(type1_value)
        print(user_team)

        types = ['Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice', 'Fighting', 'Poison', 'Ground', 'Flying', 'Psychic', 'Bug', 'Rock', 'Ghost', 'Dragon']
        user_indices = [types.index(t) for t in user_team]
        gym_indices = [types.index(t) for t in gym_leader_team]

        type_effectiveness = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Normal
            [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1],  # Fire
            [1, 1, 1, 1, 0.5, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1],  # Water
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1],  # Electric
            [1, 0.5, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1],  # Grass
            [1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 2],  # Ice
            [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 2, 2, 1, 1],  # Fighting
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],  # Poison
            [1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2],  # Ground
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Flying
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1],  # Psychic
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1],  # Bug
            [1, 1, 1, 1, 1, 1, 2, 1, 0.5, 1, 1, 1, 1, 1, 1],  # Rock
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],  # Ghost
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]   # Dragon
        ])

        base_size = 2  # 基礎圓點大小
        size_factor = 8  # 隊伍數量影響圓點大小的係數
        user_team_size = len(user_team)

        color_map = {
            2: "#FF7F50", 
            1: "#FFD700",  
            0.5: "#FFF8DC", 
            0: "#D3D3D3"   
        }

        fig = go.Figure()

        for i in range(len(user_indices)):
            for j in range(len(gym_indices)):
                # 計算對應的剋制值
                effectiveness = type_effectiveness[user_indices[i], gym_indices[j]]
                
                # 設定顏色
                color = color_map[effectiveness]
                
                # 動態調整圓點大小
                point_size = base_size + (user_team_size * size_factor)
                
                # 繪製圓點 (交換 x 和 y)
                fig.add_trace(go.Scatter(
                    x=[i], y=[j],
                    mode='markers+text',
                    marker=dict(color=color, size=point_size, opacity=0.7),
                    text=[f'{effectiveness}'],
                    textposition='middle center',
                    textfont=dict(size=min(point_size // 40 + 1, 16), color='black'),
                    showlegend=False
                ))

        # 設定標籤 (交換 x 和 y)
        fig.update_layout(
            xaxis=dict(
                tickvals=list(range(len(user_team))),
                ticktext=user_team,
                title="User's Team (Attacking Types)"
            ),
            yaxis=dict(
                tickvals=list(range(len(gym_leader_team))),
                ticktext=gym_leader_team,
                title="Gym Leader's Team (Defending Types)"
            ),
            title="Type Effectiveness Between Teams",
            showlegend=False
        )

        legend_items = [
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color_map[2]),
                name='Super Effective (2x)'
            ),
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color_map[1]),
                name='Normal Effectiveness (1x)'
            ),
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color_map[0.5]),
                name='Not Very Effective (0.5x)'
            ),
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color_map[0]),
                name='No Effect (0x)'
            )
        ]

        for item in legend_items:
            fig.add_trace(item)

        fig.update_layout(
            showlegend=True)

        return [fig]


#-----------------------------------------------------------------------------------------------

# Define the callback to update the player image based on the selected character
@app.callback(
    Output('player-image', 'children'),
    [Input('player-dropdown', 'value')]
)
def update_player_image(selected_player):
    if selected_player is None:
        return ""

    return html.Img(src=selected_player, style={'height': '200px'})


# Callback for barchart1
@app.callback(
    Output("bar-chart-1", "figure"),
    [Input('gym-leader-dropdown', 'value')]
)
def update_bar_chart(selected_leader):
    # Filter the DataFrame for the selected leader
    filtered_df = result_df[result_df["leader"] == selected_leader]
    
    # Count Pokémon types (combine type1 and type2, ignoring None)
    type_counts = (
        pd.concat([filtered_df["type1"], filtered_df["type2"]])
        .dropna()  # Remove NaN for Pokémon with no second type
        .value_counts()
        .reset_index()
    )
    type_counts.columns = ["Type", "Count"]

    # Create bar chart
    fig = px.bar(
        type_counts, 
        x="Type", 
        y="Count", 
        title=f"Pokémon Type Distribution for Leader {selected_leader}",
        labels={"Type": "Pokémon Type", "Count": "Number of Pokémon"},
        color="Type",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    # Update y-axis to use a tick step of 1
    fig.update_layout(
        yaxis=dict(
            tickmode="linear",
            dtick=1,  # Set tick step to 1
            title="Number of Pokémon"
        )
    )
    
    return fig

# Callback for barchart2
@app.callback(
    Output("bar-chart-2", "figure"),
    [Input(f'pokemon-dropdown-{i}', 'value') for i in range(6)]
)
def update_team_type_chart(*selected_pokemons):
    # Remove None values
    selected_pokemons = [p for p in selected_pokemons if p is not None]
    
    if not selected_pokemons:
        return {}
    
    # Filter DataFrame for selected Pokémon
    team_df = df[df['name'].isin(selected_pokemons)]
    
    # Count Pokémon types (combine type1 and type2, ignoring None)
    type_counts = (
        pd.concat([team_df["type1"], team_df["type2"]])
        .dropna()  # Remove NaN for Pokémon with no second type
        .value_counts()
        .reset_index()
    )
    type_counts.columns = ["Type", "Count"]

    # Create bar chart
    fig = px.bar(
        type_counts, 
        x="Type", 
        y="Count", 
        title=f"Pokémon Type Distribution for User",
        labels={"Type": "Pokémon Type", "Count": "Number of Pokémon"},
        color="Type",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    # Update y-axis to use a tick step of 1
    fig.update_layout(
        yaxis=dict(
            tickmode="linear",
            dtick=1,  # Set tick step to 1
            title="Number of Pokémon"
        )
    )
    
    return fig
if __name__ == '__main__':
    app.run_server(debug=True)