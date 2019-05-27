# imports required for the programme to run
import csv
import math
import argparse as ap
import pandas as pd

# Handling the CLI arguments
argument_parser = ap.ArgumentParser(description='Naive Bayes Classifier')
argument_parser.add_argument('-d', '--data', type=str, metavar='', required=True, help='Input file path')
argument_parser.add_argument('-o', '--output', type=str, metavar='', required=True, help='Output file path')
arguments = argument_parser.parse_args()
csv_data_path = arguments.data
output_path = arguments.output

# Reading the contents from the input location
csv_data = pd.read_csv(csv_data_path, sep='\t', header=None)
rows = csv_data.shape[0]
csv_data[0].replace(['A', 'B'], [1, 0], inplace=True)


# Method to compute the mean for every attribute wrt class label
def calc_mean(attribute,class_label):
    total = csv_data.groupby(0)[attribute].sum()[class_label]
    size = len(csv_data[csv_data[0] == class_label])
    mean = total/size
    return mean


# Method to calculate the probability of each class label
def calc_probability_of_class(class_label):
    data_with_class_label = len(csv_data[csv_data[0] == class_label])
    total_data_items = csv_data.shape[0]
    probability_of_class = data_with_class_label / total_data_items
    return probability_of_class


# Method to calculate the variance wrt attribute and class label
def calc_variance(attribute, class_label, mean):
    sum = 0
    for i in range(rows):
        if csv_data[0][i] == class_label:
            sum = sum + pow((csv_data[attribute][i] - mean), 2)
    variance = sum / (len(csv_data[csv_data[0] == class_label]) - 1)
    return variance


# Method to compute probability wrt attribute
def calc_prob_class_vs_attrib(attribute, variance, mean):
    prob_arr = []
    for i in range(rows):
        exponent = math.exp(-(math.pow(csv_data[attribute][i] - mean, 2) / (2 * variance)))
        probability = exponent / (math.sqrt(2 * math.pi * variance))
        prob_arr.append(probability)
    return prob_arr


# Method to write output to tsv file
def write_csv():
    with open(output_path, 'wt', newline='') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow([mean_class_a_attrib_1, variance_class_a_attrib_1,
                             mean_class_a_attrib_2, variance_class_a_attrib_2, probability_of_a])
        tsv_writer.writerow([mean_class_b_attrib_1, variance_class_b_attrib_1,
                             mean_class_b_attrib_2, variance_class_b_attrib_2, probability_of_b])
        tsv_writer.writerow([error])


# Method to write output to tsv file
def print_console():
        print([mean_class_a_attrib_1, variance_class_a_attrib_1,
               mean_class_a_attrib_2, variance_class_a_attrib_2, probability_of_a])
        print([mean_class_b_attrib_1, variance_class_b_attrib_1,
               mean_class_b_attrib_2, variance_class_b_attrib_2, probability_of_b])
        print([error])
        print("Output written to " + output_path + " Successfully")

# main program
# Calculating the probability of classes
probability_of_a, probability_of_b = calc_probability_of_class(1), calc_probability_of_class(0)

# Calculating the means of classes wrt attributes
mean_class_a_attrib_1, mean_class_a_attrib_2, mean_class_b_attrib_1, mean_class_b_attrib_2 = \
    calc_mean(1, 1), calc_mean(2, 1), calc_mean(1, 0), calc_mean(2, 0)

# Calculating the variance of classes wrt attributes
variance_class_a_attrib_1, variance_class_a_attrib_2, variance_class_b_attrib_1,variance_class_b_attrib_2 =\
    calc_variance(1, 1, mean_class_a_attrib_1), calc_variance(2, 1, mean_class_a_attrib_2), \
    calc_variance(1, 0, mean_class_b_attrib_1), calc_variance(2, 0, mean_class_b_attrib_2)

# Calculating the probabilities
prob_array_a_x1, prob_array_b_x1, prob_array_a_x2, prob_array_b_x2 = \
    calc_prob_class_vs_attrib(1, variance_class_a_attrib_1, mean_class_a_attrib_1).copy(),\
    calc_prob_class_vs_attrib(1, variance_class_b_attrib_1, mean_class_b_attrib_1).copy(),\
    calc_prob_class_vs_attrib(2, variance_class_a_attrib_2, mean_class_a_attrib_2).copy(),\
    calc_prob_class_vs_attrib(2, variance_class_b_attrib_2, mean_class_b_attrib_2).copy()
probability_of_a_i, probability_of_b_i = [a * b for a, b in zip(prob_array_a_x1, prob_array_a_x2)],\
                                         [a * b for a, b in zip(prob_array_b_x1, prob_array_b_x2)]

# Computing the missed predictions
error = 0
for i in range(rows):
    if (probability_of_a_i[i] > probability_of_b_i[i] and 1 or 0) != csv_data[0][i]:
        error = error + 1
# Handling Output
write_csv()
print_console()
