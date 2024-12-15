SELECT * 
FROM Gym JOIN Gym_Pokemon ON (Gym.gym_id = Gym_Pokemon.gym_id)
						JOIN Pokemon_Info ON (Gym_Pokemon.generation = Pokemon_Info.pokemon_generation
																AND Gym_Pokemon.pokemon = Pokemon_Info.name)
						JOIN Pokemon_Types ON (Pokemon_Info.pokemon_generation = Pokemon_Types.pokemon_generation
																AND Pokemon_Info.name = Pokemon_Types.name)
						JOIN Type_Effectiveness ON (Pokemon_Types.type = Type_Effectiveness.type)