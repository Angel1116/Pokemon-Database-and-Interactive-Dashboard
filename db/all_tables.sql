SELECT * 
FROM Gym JOIN Gym_Pokemon ON (Gym.gym_id = Gym_Pokemon.gym_id)
						JOIN Pokemon ON (Gym_Pokemon.generation = Pokemon.pokemon_generation
																AND Gym_Pokemon.pokemon = Pokemon.name)
						JOIN Tactics_Matrix  ON (lower(Pokemon.type1) = Tactics_Matrix.type1 
																			AND (LOWER(Pokemon.type2) = Tactics_Matrix.type2 OR (Pokemon.type2 IS NULL AND Tactics_Matrix.type2 IS NULL)))