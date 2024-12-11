# Path to Victory: Pokémon Dashboard

## Dashboard Overview

### State 1: No Input Detected
**Message:**  
Ready to start your journey? Select a Pokémon Gym Leader and build your line-up to unlock strategic insights tailored to your challenge!

**Next step:**
- Use the dropdown menus to choose the Gym Leader and build your line-up.
- Once selected, the dashboard will generate analysis and recommendations for your team-building strategy.

### State 2: Team Adjustment Recommendation
**Message:**  
Optimize your team composition! Based on the analysis of type matchups and stat comparisons, we suggest the following changes:

**Next step:**
- **Current Pokémon:** {Pikachu}
- **Replacement Reason:**  
  - **Type Disadvantage:** {Pikachu}'s {Electric} type is weak against the Gym Leader's {Ground-type} Pokémon, rendering its attacks ineffective.
  - **Stat Concern:** {Pikachu}'s Defense (35) is significantly lower than the Gym Leader’s team average Defense (65), making it vulnerable to physical attacks from the Gym Leader’s Pokémon.

### State 3: Victory-Ready Team
**Message:**  
Your current team is primed for victory! The analysis confirms your Pokémon are well-suited to defeat the Gym Leader.

**Next step:**
- Proceed confidently to battle or refine your team further if desired.

---

## Dashboard Features

### Gym Leader Type Count Bar Chart
- **Purpose:** Visualize the Pokémon type distribution in the Gym Leader's team.
- **How to Use:**
  - Use this bar chart to preliminarily build your Pokémon team with advantageous matchups to counter the most frequent type of the Gym Leader's Pokémon team.
- **Hover Information:**  
  - **Hovered Property:** {Water}  
  - **Most Effective Against:** {Electric} (2x damage)

### My Team Type Count Bar Chart
- **Purpose:** Visualize the type distribution of your Pokémon team.
- **How to Use:**
  - Start by reviewing the Gym Leader's Type Count Bar Chart to understand the dominant types in their team.
  - Adjust your own team's types to counteract the Gym Leader's advantages.

### Type Tactic Matrix
- **Purpose:** Show the effectiveness of different Pokémon types against each other. It highlights which types are strong (super-effective), neutral, or weak (not very effective) when attacking or defending.
- **How to Use:**
  - Use the matrix to select Pokémon based on type effectiveness.
  - Optimize your Pokémon's order to avoid sending out Pokémon with types that are weak against the Gym Leader’s primary types.

### Radar Chart
- **Purpose:** Compare the stats of your Pokémon team with the Gym Leader's team. It highlights the strengths and weaknesses of each Pokémon, giving you insights into how to optimize your team based on these comparisons.
- **How to Use:**
  - Compare your Pokémon’s stats with the Gym Leader’s team using the radar chart.
  - Identify strengths (higher stats) and weaknesses (lower stats) to optimize your team by reinforcing strengths and addressing weaknesses for better performance in battle.

### Stat Distribution Line Graph
- **Purpose:** Show how your Pokémon’s stats compare to the Gym Leader’s team, as well as how they rank within the general Pokémon population. This helps you understand your team’s relative performance in each stat.
- **How to Use:**
  - Track your Pokémon's stat performance using the line graph.
  - Compare each stat to the Gym Leader's average. If a Pokémon's stat is below the Gym Leader's average, consider replacing it with a Pokémon that has stronger stats in that category.
  - Use the graph to optimize your team, ensuring it is well-balanced with strong stats in critical areas for better battle performance.

---

## Get Started
1. Clone this repository.
2. Launch the dashboard and follow the steps outlined in the `Dashboard Overview`.
3. Use the visualization tools to optimize your Pokémon team.

Embark on your journey to become a Pokémon Champion!
