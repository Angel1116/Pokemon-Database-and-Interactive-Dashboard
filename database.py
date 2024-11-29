import sqlite3
import pandas as pd

# Connect to the database
conn = sqlite3.connect('pokemonStats.db')

# Execute the query and fetch the data into a DataFrame
df = pd.read_sql_query("""
SELECT * 
FROM Gym 
JOIN Gym_Pokemon ON (Gym.gym_id = Gym_Pokemon.gym_id)
JOIN Pokemon ON (Gym_Pokemon.gym_generation = Pokemon.pokemon_generation
                 AND Gym_Pokemon.pokemon = Pokemon.name)
JOIN Tactics_Matrix ON (LOWER(Pokemon.type1) = Tactics_Matrix.type1 
                        AND (LOWER(Pokemon.type2) = Tactics_Matrix.type2 OR (Pokemon.type2 IS NULL AND Tactics_Matrix.type2 IS NULL)))
LIMIT 5;
""", conn)

# Export the DataFrame to a CSV file
df.to_csv('output.csv', index=False)

# Print the DataFrame (optional)
print(df)

# Close the connection
conn.close()