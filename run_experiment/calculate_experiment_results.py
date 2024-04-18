import json
import numpy as np

f = open('experiment_results2.txt', 'r')
#print(f.read())
data = f.read()
#print(data)
data_json = json.loads(data)


dataset_results = []
for i in data_json:
    print(i)
    lists = data_json[i]

    #lists_transposed = lists.T
    transposed_data = list(zip(*lists))
    #print(transposed_data)

    one_result = {}
    one_result['dataset'] = i

    for j in range(len(transposed_data)):
        if j == len(transposed_data) - 1:
            dat = tuple(int(element) for element in transposed_data[j])
            #dat = int(transposed_data[j])
        else:
            dat = transposed_data[j]

        mean_performance = np.mean(dat)
        sample_std = np.std(dat)
        # Define z-score for 90% confidence interval (around 1.645)
        z_score = 1.645
        # Calculate the margin of error
        margin_of_error = z_score * sample_std
        # Calculate the confidence interval bounds
        lower_bound = mean_performance - margin_of_error
        upper_bound = mean_performance + margin_of_error

        res = [mean_performance, sample_std, (upper_bound, lower_bound)]
        one_result[j] = res

        # Print the results
        #print(f"Mean Performance: {mean_performance:.4f}")
        #print(f"Standard Deviation (Optional): {sample_std:.4f}")  # Replace with your estimate if unavailable
        #print(f"90% Confidence Interval: [{lower_bound:.4f}, {upper_bound:.4f}]")

    dataset_results.append(one_result)

for k in dataset_results:
    print(k)
#print(dataset_results)

'''
    averages = []
    for j in range(len(lists[0])):
        tot_sum = 0
        for sublist in lists:
            value = float(sublist[j])
            #tot_sum += int(sublist[j])
            tot_sum += value
        avg = tot_sum / len(lists)
        averages.append(avg)

    dataset_results.append(averages)
'''
#print(dataset_results)