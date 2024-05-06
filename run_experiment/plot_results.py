
import json

import matplotlib.pyplot as plt
import numpy as np

#f = open('experiment_plots_5clients.txt', 'r')
#data = f.read()
#data_json = json.loads(data)


def plot_selected_regression_data(dictionary, entry):
    data = dictionary[entry]

    nr = len(data[0])
    x = list(range(nr))
    y1 = data[0]
    y2 = data[1]

    plt.scatter(x, y1, color='green', label='green')
    plt.scatter(x, y2, color='red', label='red')
    plt.legend()
    plt.show()
    return False


def read_data_from_file():
    with open('experiment_plots_testing.txt', 'r') as file:
        contents = file.read()

    #lines = contents.split('\n')
    lines = contents.split(']\n')
    # remove last empty row
    lines = lines[:-1]

    data = {}
    key = ''
    run = 0
    # Process each line
    for line in lines:
        # Split the line based on the colon (":") separator

        key, value = line.split(':')
        # Strip the whitespace from the key and value
        key = key.strip()
        value = value.strip()
        # Remove the brackets and split the values based on comma and space (", ")

        sublists = value.split('],[')
        #values = value.strip('[]').split(', ')
        lists_ = []
        for _list in sublists:
            _list = _list.replace(",", "")
            values = _list.strip('[]').split()
            # Convert the string values to integers or floats
            values = [int(v) if v.isdigit() else float(v) for v in values]
            # Store the data in the dictionary
            lists_.append(values)

        data[f'{key}{run}'] = (lists_[0], lists_[1])
        run += 1
        #data[key] = values

    # Print the data
    for key, value in data.items():
        print(key, ":", value)

    return data

data = read_data_from_file()

# plot a selected dictionary entry
entry = 'wine_quality5'
plot_selected_regression_data(data, entry)