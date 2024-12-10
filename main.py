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
import math

# Load the Gym.csv and Gym_Pokemon.csv files into DataFrames
gym_df = pd.read_csv('Gym.csv')
gym_pokemon_df = pd.read_csv('Gym_Pokemon.csv')
# Load the Pokemon.csv file into a DataFrame
df = pd.read_csv('Pokemon.csv', encoding='latin1')

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

color_list = ['#FFE5B4', '#FFD700', '#FFB700','#2FBDBD','#80DAEB', '#6798C0']

counter_data = {
    'Type': ['Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice', 'Fighting', 'Poison', 'Ground', 
             'Flying', 'Psychic', 'Bug', 'Rock', 'Ghost', 'Dragon'],
    'Counters': ['Fighting', 'Water, Rock', 'Electric, Grass', 'Ground', 'Fire, Flying', 'Fighting, Fire, Rock', 
                 'Flying, Psychic', 'Ground, Psychic', 'Water, Ice, Grass', 'Electric, Ice, Rock', 
                 'Bug, Ghost, Dark', 'Flying, Rock, Fire', 'Fighting, Ground, Steel, Water, Grass', 
                 'Ghost, Dark', 'Dragon, Fairy']
}
counter_df = pd.DataFrame(counter_data)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([

    # Title
    html.Div([

        html.Div([
            html.H2("Pokémon Gym Leader Battle Simulator", 
                    style={
                        'textAlign': 'left',
                        'fontFamily': 'Calibri',
                        'fontSize': '36px',
                        'marginBottom': '10px',
                        'color': '#2F4F4F'  # Dark slate gray color
                    }
            ),
            html.P("The dashboard helps players make informed decisions when optimizing their Pokémon teams to take on specific Gym Leaders.", 
                    style={
                        'textAlign': 'left',
                        'fontFamily': 'Calibri',
                        'fontSize': '16px',
                        'marginBottom': '20px',
                        'color': '#696969'  # Dimgray color
                    }
            )
        ], style={
            'display': 'inline-block',
            'vertical-align': 'top',
            'width': '65%',
            'box-sizing': 'border-box',
            'font-family': 'Calibri'  # 設定 Calibri 字體
        }),

        # Instructions Section
        html.Div([
            html.H2("Welcome, Trainer!", 
                style={
                    'margin-bottom': '10px',
                    'font-family': 'Calibri',
                    'color': '#2F4F4F',
                    'font-size': '24px'
                }),
            html.Div(id='instruction-text', style={
                'text-align': 'left',
                'line-height': '1.6',
                'padding': '20px',
                'background-color': 'rgba(255, 255, 255, 0.7)',
                'border-radius': '10px',
                'border': '2px solid #e0e0e0',
                'box-shadow': '0 2px 4px rgba(0,0,0,0.05)',
                'font-family': 'Calibri',
                'font-size': '18px',
                'color': '#444',
                'margin': '10px 0'
            })
        ], style={
            'width': '30%',
            'display': 'inline-block',
            'vertical-align': 'top',
            'marginLeft': 'auto',
            'marginTop': '0px',
            'padding': '10px',
            'float': 'right'
        })
    ], style={
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'flex-start',
        'marginBottom': '30px',
        'width': '100%'
    }),

    # Top row with Gym Leader and their Pokémon
    html.Div([
        # Gym Leader Dropdown and Info
        html.Div([
            dcc.Dropdown(
                id='gym-leader-dropdown',
                options=[{'label': leader, 'value': leader} for leader in merged_df['leader'].unique()],
                placeholder="Select a Gym Leader",
                style={'width': '150px', 'display': 'inline-block','font-family': 'Calibri'}  # 設定 Calibri 字體
            ),
            html.Div(id='leader-info', style={'margin-top': '20px'})
        ], style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top','font-family': 'Calibri','height': '80px','margin-left': '20px','margin-top': '50px',}),  # 設定 Calibri 字體

        # Gym Leader's Pokémon Images
        html.Div(id='pokemon-images', style={
            'width': '40%',
            'display': 'grid',
            'grid-template-columns': 'repeat(3, 1fr)',
            'grid-gap': '2px',
            'justify-content': 'center',
            'font-family': 'Calibri',  # 設定 Calibri 字體
            'height': '30px',
            'margin-top': '80px',
            'margin-left': '50px',
            'margin-right': '55px',
            
        }),

        #bar chart1
        html.Div([
            dcc.Graph(id="bar-chart-1",style={'height':'400px','width': '95%'}),
            html.Div("Use this to build your team to counter the gym leader's most common type.",
                style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center','font-size': '14px', 'color': 'gray', 'font-family': 'Calibri','white-space': 'normal'}) 
            ], ), 
        ], style={
        'width': '100%', 'display': 'flex','height':'450px',
        'background-color': '#EDF2F4',
        'border-radius': '10px',
        'margin-top': '20px'}),
    


    # Second row with Team Pokémon
    html.Div([
        html.Div([
            dcc.Dropdown(
                id='player-dropdown',
                options=[{'label': img.split('/')[-1].split('.')[0], 'value': img} for img in player_images],
                placeholder="Select Your Character",
                style={'width': '150px', 'display': 'inline-block', 'font-family': 'Calibri'}  # 設定 Calibri 字體
            ),
            html.Div(id='player-image', style={'margin-top': '0px',  'height': '290px'})
        ], style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '20px', 'font-family': 'Calibri'}),  # 設定 Calibri 字體


        # Pokémon Dropdowns and Images
        html.Div([
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id=f'pokemon-dropdown-{i}',
                        options=[{'label': pokemon, 'value': pokemon} for pokemon in df['name']],
                        placeholder=f'Select Pokémon {i+1}',
                        style={'width': '90%', 'margin-top': '0px', 'font-family': 'Calibri','align-items': 'flex-start'}  # 設定 Calibri 字體
                    ),
                    html.Div(id=f'image-container-{i}', style={
                        'width': '100px',
                        'height': '110px',
                        'margin-top': '5px',
                        'display': 'flex',
                        'flex-direction': 'column',
                        'justify-content': 'center',
                        'align-items': 'center',
                        'font-family': 'Calibri'  # 設定 Calibri 字體
                    }),
                     # Pokémon Name
                    html.Div(id=f'name-container-{i}', style={
                    'text-align': 'center',
                    'font-weight': 'bold',
                    'margin-top': '5px',
                    'font-family': 'Calibri'
                }),
                ], style={
                    'width': '200px',
                    'padding': '5px',
                    'box-sizing': 'border-box',
                    'margin-top': '0px',
                    'font-family': 'Calibri',  # 設定 Calibri 字體
                    'margin-left': '0',  # Remove left margin for the container
                    'padding-left': '0',
                    'align-items': 'center'
                }) for i in range(6)
            ], style={                       #裝寶可夢圖片的大容器
                'width': '100%',
                'display': 'grid',
                'grid-template-columns': 'repeat(3, 1fr)', # 3 columns of equal width
                'grid-gap': '10px',
                'justify-content': 'center',
                'align-items': 'center',
                'font-family': 'Calibri',
                'margin-left': '0 auto',
                'padding': '0px'
            })
        ], style={'width': '40%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-top': '10px','padding': '0px', 'font-family': 'Calibri'}),  # 設定 Calibri 字體

        #bar chart2
        html.Div([
            dcc.Graph(id="bar-chart-2",style={'height':'400px','width': '95%'}),
            html.Div("Check the Gym Leader's Type Bar Chart to identify dominant types and adjust your team.",
                style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center','font-size': '14px', 'color': 'gray', 'font-family': 'Calibri','white-space': 'normal'}) 
        ], ),  
    ], style={
        'display': 'flex',
        'width': '100%',
        'justify-content': 'space-between',  # Adds spacing between the charts
        'align-items': 'center',  # Aligns charts vertically in case of differing heights
        'font-family': 'Calibri',  # 設定 Calibri 字體
        'background-color': '#EDF2F4',
        'border-radius': '10px',
        'margin-top': '20px',
        'height':'450px'
    }),



    html.Div([
        # Heatmap
        html.Div([
            dcc.Graph(id='heatmap', style={'width': '100%', 'height': '390px'}),
            html.Div("Show the effectiveness of different Pokémon types against each other. It highlights which types are strong (super-effective), neutral, or weak (not very effective) when attacking or defending.",
                style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center','font-size': '14px', 'color': 'gray', 'font-family': 'Calibri'}) 
        ], style={'width': '100%', 'margin-top': '10px', 'align-items': 'flex-start', 'justify-content': 'center','background-color': 'transparent','padding-left': '20px',
                'background-color': '#EDF2F4',
                'border-radius': '10px',
                'margin-top': '20px'}),

        # Radar Chart
        html.Div([
            dcc.Graph(id='radar-chart', style={'width': '90%', 'height': '390px', 'margin': 'auto'}),
            html.Div("Compare the stats of your Pokémon team with the Gym Leader's team. It highlights the strengths and weaknesses of each Pokémon, giving you insights into how to optimize your team based on these comparisons.",
                style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center','font-size': '14px', 'color': 'gray', 'font-family': 'Calibri','padding-left': '20px',  'padding-right': '20px' }) 
        ], style={'width': '95%', 'margin-top': '10px', 'align-items': 'flex-start','justify-content': 'center', 'background-color': 'transparent','margin-left': '20px',
                'background-color': '#EDF2F4',
                'border-radius': '10px',
                'margin-top': '20px'}),

        # Distribution Line
        html.Div([
            dcc.Graph(id='distribution-line', style={'width': '100%', 'height': '390px'}),
            html.Div("Show how your Pokémon's stats compare to the Gym Leader's team, as well as how they rank within the general Pokémon population. This helps you understand your team's relative performance in each stat.",
                style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center','font-size': '14px', 'color': 'gray', 'font-family': 'Calibri','padding-left': '20px',  'padding-right': '20px' }) 
        ], style={'width': '100%', 'margin-top': '10px',  'align-items': 'flex-start', 'justify-content': 'center','background-color': 'transparent','margin-left': '20px',
                'background-color': '#EDF2F4',
                'border-radius': '10px',
                'margin-top': '20px'}),
    ], style={
        'display': 'flex',
        'width': '100%',
        'justify-content': 'space-between',  # Adds spacing between the charts
        'align-items': 'center',  # Aligns charts vertically in case of differing heights
        'font-family': 'Calibri' 
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
        html.Img(src=leader_image_path, style={'height': '120px'}),
        html.H3(selected_leader)
    ])

    # Create the Pokémon images div with names and types
    pokemon_images = []  # Ensure this is inside the function
    for _, row in leader_df.iterrows():
        pokemon_images.append(
            html.Div([
                # Pokémon image
                html.Img(
                    src=row['pokemon_image_path'],
                    style={'max-width': '100%', 'max-height': '150px'}
                ),

                # Pokémon types
                html.Div(f"Type: {row['type1']} / {row['type2'] if pd.notna(row['type2']) else 'None'}",
                         style={'text-align': 'center', 'font-size': '12px', 'color': 'gray'})
      
            ], style={
                'display': 'flex',
                'flex-direction': 'column',
                'align-items': 'center',
                'justify-content': 'center',
                'margin': '10px'
            })
        )
    

    return leader_info, pokemon_images


# Define the callback to update the images and radar chart based on the selected Pokémon and gym leader
@app.callback(
    [Output(f'image-container-{i}', 'children') for i in range(6)] +
    [Output('radar-chart', 'figure')],
    [Input(f'pokemon-dropdown-{i}', 'value') for i in range(6)] +
    [Input('gym-leader-dropdown', 'value')]
)
def update_content(selected_pokemon_0, selected_pokemon_1, selected_pokemon_2, 
                   selected_pokemon_3, selected_pokemon_4, selected_pokemon_5, 
                   selected_leader):
    # Create a list of selected Pokémon for easier iteration
    selected_pokemons = [
        selected_pokemon_0, selected_pokemon_1, selected_pokemon_2, 
        selected_pokemon_3, selected_pokemon_4, selected_pokemon_5
    ]

    print("Selected Pokémon:", selected_pokemons)
    
    
    images = []
    team_stats = {
        'HP': 0,
        'attack': 0,
        'defense': 0,
        'special_attack': 0,
        'special_defense': 0,
        'speed': 0
    }
    leader_stats = {
        'HP': 0,
        'attack': 0,
        'defense': 0,
        'special_attack': 0,
        'special_defense': 0,
        'speed': 0
    }
    
    # Calculate team stats
    for selected_pokemon in selected_pokemons:
        if selected_pokemon is not None:
            # Get the image path
            image_path = df[df['name'] == selected_pokemon]['image_path'].values[0]
            pokemon_types = df[df['name'] == selected_pokemon][['type1', 'type2']].iloc[0]
            images.append(html.Div([
                html.Img(src=image_path, style={'max-width': '100%', 'max-height': '150px'}),
                html.Div(f"Type: {pokemon_types['type1']} / {pokemon_types['type2'] if pd.notna(pokemon_types['type2']) else 'None'}",
                         style={'text-align': 'center', 'font-size': '12px', 'color': 'gray'})
            ], style={
                'display': 'flex',
                'flex-direction': 'column',
                'align-items': 'center',
                'justify-content': 'center',
                'margin': '10px'
            }))

            # Get the stats for the selected Pokémon
            pokemon_stats = df[df['name'] == selected_pokemon].iloc[0]
            individual_stats = {
                'HP': pokemon_stats['HP'],
                'attack': pokemon_stats['attack'],
                'defense': pokemon_stats['defense'],
                'special_attack': pokemon_stats['special_attack'],
                'special_defense': pokemon_stats['special_defense'],
                'speed': pokemon_stats['speed']
            }

            # Add individual pokemon stats to team stats
            for stat_name in team_stats.keys():
                team_stats[stat_name] += individual_stats[stat_name]
        else:
            images.append(html.Div())

    # Calculate leader stats
    if selected_leader is not None:
        leader_df = merged_df[merged_df['leader'] == selected_leader]
        for _, row in leader_df.iterrows():
            pokemon_stats = row[['HP', 'attack', 'defense', 'special_attack', 'special_defense', 'speed']]
            for stat_name in leader_stats.keys():
                leader_stats[stat_name] += pokemon_stats[stat_name]

    # Pad with empty divs if fewer than 6 Pokémon are selected
    while len(images) < 6:
        images.append(html.Div())

    # Create the radar chart with both team stats and leader stats
    radar_chart = {
        'data': [
            {
                'type': 'scatterpolar',
                'r': list(team_stats.values()),
                'theta': list(team_stats.keys()),
                'fill': 'toself',
                'name': 'Team Total',
                'marker': {'color': '#FFD700'},
                'line': {'color': '#FFD700'}
            },
            {
                'type': 'scatterpolar',
                'r': list(leader_stats.values()),
                'theta': list(leader_stats.keys()),
                'fill': 'toself',
                'name': f"{selected_leader}'s Team Total",
                'marker': {'color': '#6798C0'},
                'line': {'color': '#6798C0'}
            }
        ],
        'layout': {
            'polar': {
                'radialaxis': {
                    'visible': True,
                    'range': [0, max(max(team_stats.values()), max(leader_stats.values())) + 100]  # Adjust the range as needed
                },
                'bgcolor': 'transparent'
            },
            'showlegend': True,
            'title': 'Team Total Stats',
            'paper_bgcolor': 'transparent',  # Add this line
            'plot_bgcolor': 'transparent'    # Add this line
        }
    }

    return images + [radar_chart]

#Yi-Syuan-----------------------------------------------------------------------------------------------
# Define the callback to update the distribution line based on the selected Pokémon
@app.callback(
    [Output('distribution-line', 'figure')],
    [Input('gym-leader-dropdown', 'value')] + 
    [Input(f'pokemon-dropdown-{i}', 'value') for i in range(6)]
)
def update_content(selected_leader, *selected_pokemons):
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
    colors = {}

    if selected_leader is None:
        return [{
            "data": [],
            "layout": {
                "title": "Stats Distribution of your team",
                "plot_bgcolor": "rgba(0,0,0,0)",
                "paper_bgcolor":'rgba(0, 0, 0, 0)'
            }
        }]
    
    if selected_leader is not None:
        gym_df = merged_df[merged_df['leader'] == selected_leader][['HP', 'attack', 'defense', 'special_attack', 'special_defense', 'speed']]
        mean_gym = gym_df.mean().tolist()
        columns = ['HP', 'attack', 'defense', 'special_attack', 'special_defense', 'speed']
        pr_gym = []
        for i, col in enumerate(columns):
            pr = round((df[col] <= mean_gym[i]).mean() * 100, 0)
            pr_gym.append(pr)
        pr_gym_df = pd.DataFrame({'pr': pr_gym}, index=columns)

    for idx, selected_pokemon in enumerate(selected_pokemons):
        if selected_pokemon is not None:
            pokemon_stats = df[df['name'] == selected_pokemon][['HP', 'attack', 'defense', 'special_attack', 'special_defense', 'speed']].iloc[0]
            pr = [
                round((df[stat] <= pokemon_stats[stat]).mean() * 100,0)
                for stat in pokemon_stats.index
            ]
            dist_df.loc[selected_pokemon] = pr
            color = color_list[idx % len(color_list)]
            colors[selected_pokemon] = color

    stats = df[['HP', 'attack', 'defense', 'special_attack', 'special_defense', 'speed']].agg(['min', 'max', 'mean']).transpose()
    print(stats)
    fig = go.Figure()
    used_x_positions = {col: [] for col in dist_df.columns}

    for col in dist_df.columns:
        one_pr_gym = pr_gym_df.loc[col, 'pr']

        fig.add_trace(go.Scatter(
            x=[0, 100],
            y=[col, col],
            mode='lines',
            line=dict(color='white', width=10),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[0, 100],  # Start and end points
            y=[col, col],  # Same y-coordinate as the line
            mode='markers',
            marker=dict(
                size=10,  # Match the line width
                color='white',
                symbol='circle'  # Use circular markers
            ),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[one_pr_gym],
            y=[col],
            mode='markers+text',
            text='Gym',
            textposition='bottom center',
            hovertext=[f'Average of gym:{one_pr_gym}'],
            hoverinfo='text',
            marker=dict(color='#E23D28', size=13, symbol='circle'),
            showlegend=False
        ))

    for col in dist_df.columns:
        for index, value in zip(dist_df.index, dist_df[col]):
            offset = 0
            while (value + offset in used_x_positions[col]) or (value + offset == pr_gym_df.loc[col, 'pr']):
                offset += 1.5
            
            final_x = value + offset
            used_x_positions[col].append(final_x)
            if dist_df.columns[0] == col:
                fig.add_trace(go.Scatter(
                x=[final_x], 
                y=[col],  
                mode='markers+text',
                marker=dict(color=colors[index], size=13, symbol='circle'),  
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
                    marker=dict(color=colors[index], size=13, symbol='circle'),
                    hovertext=[f'{index}: {value}'],
                    hoverinfo='text',
                    showlegend=False
                ))

    fig.update_layout(
        title={
        "text": "Stats Distribution of your team",  # 設定標題內容
        "x": 0.5,  # 水平置中 (0 表示最左，1 表示最右)
        "xanchor": "center",  # 確保標題的錨點在中心
        "yanchor": "top"  # 確保標題的垂直錨點在頂端
        },
        xaxis=dict(zeroline=False,  showgrid=False),
        yaxis=dict(showticklabels=True, tickvals=list(dist_df.columns), ticktext=dist_df.columns, showgrid=False),
        height=400,
        width=500,
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(x=1, y=1, traceorder="normal", bgcolor="rgba(255, 255, 255, 0)"),
        font=dict(family="Calibri", size=14),
        paper_bgcolor='rgba(0, 0, 0, 0)')

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
                "title": "Type Effectiveness Between Teams",
                "plot_bgcolor": "rgba(0,0,0,0)",
                "paper_bgcolor":'rgba(0, 0, 0, 0)'
            }
        }]

    else:
        types = ['Normal', 'Fire', 'Water', 'Electric', 'Grass', 'Ice', 'Fighting', 'Poison', 'Ground', 'Flying', 'Psychic', 'Bug', 'Rock', 'Ghost', 'Dragon']
        raw_gym_leader_team = list(set(merged_df.loc[merged_df['leader'] == selected_leader, 'type1']).union(merged_df.loc[merged_df['leader'] == selected_leader, 'type2']))
        gym_leader_team = [x for x in raw_gym_leader_team if x in types]
        print(gym_leader_team)

        user_team = []
        for selected_pokemon in selected_pokemons:
            row = df[df['name'] == selected_pokemon]
            if not row.empty:
                type1_value = row['type1'].iloc[0]
                user_team.append(type1_value)
                type2_value = row['type2'].iloc[0]
                user_team.append(type2_value)
        user_team = list(set(user_team))
        user_team = [x for x in user_team if x in types]
        print(user_team)

        
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

        base_size = 20  

        color_map = {
            2: '#6798C0', 
            1: "#FFD700",  
            0.5: "#FFF8DC", 
            0: "#D3D3D3"   
        }

        fig = go.Figure()

        for i in range(len(user_indices)):
            for j in range(len(gym_indices)):
                # 計算對應的剋制值
                effectiveness = type_effectiveness[user_indices[i], gym_indices[j]]
                color = color_map[effectiveness]
                point_size = base_size
                
                # 繪製圓點 (交換 x 和 y)
                fig.add_trace(go.Scatter(
                    x=[i], y=[j],
                    mode='markers+text',
                    marker=dict(color=color, size=point_size, opacity=0.7),
                    text=[f'{effectiveness}'],
                    textposition='middle center',
                    textfont=dict(size=6, color='black'),
                    showlegend=False
                ))

        # 設定標籤 (交換 x 和 y)
        fig.update_layout(
            xaxis=dict(
                tickvals=list(range(len(user_team))),
                ticktext=user_team,
                title="User's Team (Attacking Types)",
                showgrid=False
            ),
            yaxis=dict(
                tickvals=list(range(len(gym_leader_team))),
                ticktext=gym_leader_team,
                title="Gym Leader's Team (Defending Types)",
                showgrid=False
            ),
            title={
            "text": "Type Effectiveness Between Teams",  # 設定標題內容
            "x": 0.5,  # 水平置中 (0 表示最左，1 表示最右)
            "xanchor": "center",  # 確保標題的錨點在中心
            "yanchor": "top"  # 確保標題的垂直錨點在頂端
            },
            plot_bgcolor='rgba(0, 0, 0, 0)',  
            paper_bgcolor='rgba(0, 0, 0, 0)',
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
            showlegend=True,
            plot_bgcolor='rgba(0, 0, 0, 0)',  # 設定圖表區域的背景透明
            paper_bgcolor='rgba(0, 0, 0, 0)',
            xaxis=dict(showgrid=False), 
            yaxis=dict(showgrid=False) )

        return [fig]

#-----------------------------------------------------------------------------------------------
# Callback for updating the instructions
@app.callback(
    Output('instruction-text', 'children'),
    [Input('gym-leader-dropdown', 'value')] +
    [Input(f'pokemon-dropdown-{i}', 'value') for i in range(6)] +
    [Input('heatmap', 'figure')]
)
def update_instructions(selected_leader, *args):
    selected_pokemons = args[:6]
    heatmap_data = args[6]

    if not selected_leader:
        return [
            "🎮 Please select a Gym Leader to begin!",
            html.Br(),
            html.Br(),
            "Choose your opponent from the dropdown menu above."
        ]

    if not any(selected_pokemons):
        return [
            "🔍 Time to build your team!",
            html.Br(),
            html.Br(),
            f"Select your Pokémon to challenge {selected_leader}."
        ]

    # Get the number of Pokémon the leader has
    leader_pokemon_count = merged_df[merged_df['leader'] == selected_leader]['pokemon'].nunique()

    # Check if the user has selected the same number of Pokémon as the leader
    selected_pokemon_count = len([pokemon for pokemon in selected_pokemons if pokemon is not None])
    if selected_pokemon_count < leader_pokemon_count:
        return [
            "⚠️ You need to select more Pokémon!",
            html.Br(),
            html.Br(),
            f"Select {leader_pokemon_count - selected_pokemon_count} more Pokémon to match the leader's team."
        ]
    
    # Check heatmap data for 'No effect' and 'Not Very Effective' values
    has_ineffective_matchup = False
    for trace in heatmap_data['data']:
        if 'text' in trace:
            for text in trace['text']:
                if text == '0' or text == '0.5':
                    has_ineffective_matchup = True
                    break
        if has_ineffective_matchup:
            break
                    
    if has_ineffective_matchup:
        return [
            "⚠️ Some matchups need attention!",
            html.Br(),
            html.Br(),
            "Your team has Pokémon with 'Not Very Effective' or 'No Effect' against the Gym Leader's team. Consider adjusting your team composition."
        ]   

    return [
        "🏆 Ready to battle!",
        html.Br(),
        html.Br(),
        "Your team is set. Check the radar chart to see how your stats compare!"
    ]


#-----------------------------------------------------------------------------------------------

# Define the callback to update the player image based on the selected character
@app.callback(
    Output('player-image', 'children'),
    [Input('player-dropdown', 'value')]
)
def update_player_image(selected_player):
    if selected_player is None:
        return ""

    return html.Img(src=selected_player, style={'height': '120px'})


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
    
    # Add counters information to the type_counts DataFrame
    type_counts['Counters'] = type_counts['Type'].map(lambda x: counter_df[counter_df['Type'] == x]['Counters'].values[0])
    
    # Create bar chart
    fig = px.bar(
        type_counts, 
        x="Type", 
        y="Count", 
        title=f"Pokémon Type Distribution for Leader",
        labels={"Type": "Pokémon Type", "Count": "Number of Pokémon"},
        color="Type",
        color_discrete_sequence=['#6798C0'],
        hover_data={'Type': True, 'Count': True, 'Counters': True}  # 添加悬停时显示的字段
    )

    
    # Add custom hover text (if you want more customization)
    fig.update_traces(
        hovertemplate=(
            'Type: %{x}<br>'  
            'Count: %{y}<br>'  
            'Counters: %{customdata[0]}<br>'  # 最怕的屬性
            '<extra></extra>'  
        )
    )
    
    # Update y-axis to use a tick step of 1
    fig.update_layout(
        yaxis=dict(
            tickmode="linear",
            dtick=1,  # Set tick step to 1
            title="Number of Pokémon"
        ),
        plot_bgcolor='rgba(0,0,0,0)',  # 圖表背景透明
        paper_bgcolor='rgba(0,0,0,0)' # 整體背景透明
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

    # Create a mapping of Type to Pokémon names
    type_to_pokemon = {}
    for t in type_counts['Type']:
        # Get Pokémon names for the current type
        pokemon_names = team_df[team_df['type1'] == t]['name'].tolist() + team_df[team_df['type2'] == t]['name'].tolist()
        type_to_pokemon[t] = ', '.join(set(pokemon_names))  # Use set to avoid duplicates

    # Add custom hovertext to the type_counts DataFrame
    type_counts['Pokemon Names'] = type_counts['Type'].map(type_to_pokemon)

    # Create bar chart
    fig = px.bar(
        type_counts, 
        x="Type", 
        y="Count", 
        title=f"Pokémon Type Distribution for User",
        labels={"Type": "Pokémon Type", "Count": "Number of Pokémon"},
        color="Type",
        color_discrete_sequence=['#FFD700'],
        hover_data={'Type': False, 'Count': True, 'Pokemon Names': True}  # 显示宝可梦名称
    )
    
    # Add custom hover text (hovertemplate) to show Pokémon names
    fig.update_traces(
        hovertemplate=(
            'Type: %{x}<br>'  # 显示类别名称
            'Count: %{y}<br>'  # 显示数量
            'Pokémon Names: %{customdata[0]}<br>'  # 显示对应类型的宝可梦名称
            '<extra></extra>'  # 隐藏默认的额外信息
        )
    )

    # Update y-axis to use a tick step of 1
    fig.update_layout(
        yaxis=dict(
            tickmode="linear",
            dtick=1,  # Set tick step to 1
            title="Number of Pokémon"
        ),
        plot_bgcolor='rgba(0,0,0,0)',  # 图表背景透明
        paper_bgcolor='rgba(0,0,0,0)' # 整体背景透明
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)