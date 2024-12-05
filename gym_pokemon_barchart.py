import sqlite3
import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output


# Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Gym Pokémon Type Distribution", style={'textAlign': 'center'}),
    html.Div([
        dcc.Dropdown(
            id="leader-dropdown",
            options=leader_options,
            value=leader_options[0]["value"],  
            placeholder="Select a Gym Leader",
            style={'width': '50%', 'margin': '0 auto'}
        )
    ], style={'textAlign': 'center'}),
    html.Div([
        dcc.Graph(id="type-bar-chart")
    ])
])

# Callback for interactivity
@app.callback(
    Output("type-bar-chart", "figure"),
    [Input("leader-dropdown", "value")]
)
def update_bar_chart(selected_leader):
    # Filter the DataFrame for the selected leader
    filtered_df = df[df["leader"] == selected_leader]
    
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

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
