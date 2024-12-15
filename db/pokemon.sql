SELECT * 
FROM Pokemon_Info JOIN Pokemon_Types ON (Pokemon_Info.pokemon_generation = Pokemon_Types.pokemon_generation
										AND Pokemon_Info.name = Pokemon_Types.name)