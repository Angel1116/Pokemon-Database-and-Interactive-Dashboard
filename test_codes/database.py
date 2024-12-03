import pandas as pd
import sqlite3

# Connect to the database
conn = sqlite3.connect('pokemonStats_1127.db')

# Fetch the Gym table data into a DataFrame
gym_df = pd.read_sql_query("SELECT * FROM Gym", conn)
# Export the Gym DataFrame to a CSV file
gym_df.to_csv('Gym.csv', index=False)

# Fetch the Pokemon table data into a DataFrame
pokemon_df = pd.read_sql_query("SELECT * FROM Pokemon", conn)
# Export the Pokemon DataFrame to a CSV file
pokemon_df.to_csv('Pokemon.csv', index=False)

# Fetch the Gym_Pokemon table data into a DataFrame
gym_pokemon_df = pd.read_sql_query("SELECT * FROM Gym_Pokemon", conn)
# Export the Gym_Pokemon DataFrame to a CSV file
gym_pokemon_df.to_csv('Gym_Pokemon.csv', index=False)

# Close the connection
conn.close()