CREATE VIEW GymPokemonDetails AS
SELECT 
    Gym_Pokemon.gym_id, 
    Gym_Pokemon.generation, 
    Gym_Pokemon.pokemon, 
    Gym.leader, 
    Pokemon.type1, 
    Pokemon.type2
FROM Gym_Pokemon
JOIN Gym ON Gym_Pokemon.gym_id = Gym.gym_id
JOIN Pokemon ON Gym_Pokemon.pokemon = Pokemon.name;

