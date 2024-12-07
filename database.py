import pandas as pd


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

print(merged_df.info())
print(merged_df.head())