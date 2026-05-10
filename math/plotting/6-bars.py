#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

# Data setup
people = ['Farrah', 'Fred', 'Felicia']
fruits = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
width = 0.5

# Plotting the stacked bars
fig, ax = plt.subplots()

# Starting height for stacking
bottom = np.zeros(len(people))

for i in range(len(fruit)):
    ax.bar(people, fruit[i], width, bottom=bottom, color=colors[i], label=fruits[i])
    bottom += fruit[i]

# Formatting the axis and title
ax.set_ylabel('Quantity of Fruit')
ax.set_title('Number of Fruit per Person')
ax.set_ylim(0, 80)
ax.set_yticks(np.arange(0, 81, 10))

# Adding the legend
ax.legend()

plt.show()
