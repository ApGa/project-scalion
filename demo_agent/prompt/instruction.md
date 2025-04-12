# Instruction
You are a proficient computer user on navigating map websites. Based on the provided task instruction and action trajectory, your task is to summarize the useful strategies to query the map website, such that the user can locate the correct locations and find the right information.

## Example
Instruction: Tell me the coordinates of Apple Store near Pitt in DD format
Action trajectory:
```
fill('140', 'Apple Store near Upitt')  # Entered "Apple Store near Upitt" in the search field to search for store location.
click('143')  # Clicked the "Go" button to initiate the search for Apple Store near Upitt.
click('450')  # Searched for "Apple Store near Upitt" but found no matching results.
click('450')  # Zoomed in on the map to look for nearby Apple Store locations.
fill('140', 'Apple Store')  # Entered "Apple Store" in the search bar to perform a broader search.
click('143')  # Clicked the "Go" button to perform a search for "Apple Store".
fill('140', 'Apple Store Pittsburgh')  # Entered "Apple Store" in the search bar to perform a broader search.
click('143')  # Clicked the "Go" button to perform a search for "Apple Store".
```

Search Strategy:
1. Avoid or expand abbreviations in the search query, e.g., remove "Upitt" and only search for "Apple Store".
2. Infer city names from the context, e.g., "Pittsburgh" from "Upitt"; and add them to the search query, e.g, "Apple Store Pittsburgh".
