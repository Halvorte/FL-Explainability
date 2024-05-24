import matplotlib.pyplot as plt

# Define data points for each line
line1_data = [(0, 1), (0.5, 0)]
line2_data = [(0, 0), (0.5, 1), (1, 0)]
line3_data = [(0.5, 0), (1, 1)]

# Create the plot
plt.figure()

# Plot each line with different styles and labels
plt.plot(*zip(*line1_data), label='LOW', marker='o', color='b')  # Unpack data points for line1
plt.plot(*zip(*line2_data), label='MEDIUM', marker='s', color='g')  # Unpack data points for line2
plt.plot(*zip(*line3_data), label='HIGH', marker='^', color='r')  # Unpack data points for line3

# Set axis limits
plt.xlim(0, 1)
plt.ylim(0, 1)

# Add labels and title
plt.xlabel('Fuzzy value')
plt.ylabel('Membership value')
plt.title('Fuzzy sets')

# Add legend
plt.legend()

# Show the plot
plt.grid(True)

plt.savefig('images/fuzzy_membership_plot.png')
plt.show()
