import pandas as pd
import numpy as np
import csv
import math
import sys
import argparse

parser = argparse.ArgumentParser(description='Naive Bayes')
parser.add_argument('-d', '--data', type=str, metavar='', required=True, help='Enter input file path')
parser.add_argument('-o', '--output', type=str, metavar='', required=True, help='Enter output file path')

args = parser.parse_args()
data_path = args.data
output_path = args.output

data = pd.read_csv(data_path, sep='\t', header=None)
data[0].replace(['A', 'B'], [1, 0], inplace=True)
rows = data.shape[0]

p_A = len(data[data[0] == 1]) / data.shape[0]
p_B = len(data[data[0] == 0]) / data.shape[0]

mean_a_x_1 = data.groupby(0)[1].sum()[1] / len(data[data[0] == 1])
mean_a_x_2 = data.groupby(0)[2].sum()[1] / len(data[data[0] == 1])
mean_b_x_1 = data.groupby(0)[1].sum()[0] / len(data[data[0] == 0])
mean_b_x_2 = data.groupby(0)[2].sum()[0] / len(data[data[0] == 0])
print(mean_a_x_1,mean_a_x_2,mean_b_x_1,mean_b_x_2)
sum_a_x_1 = 0
for i in range(rows):
    if data[0][i] == 1:
        sum_a_x_1 = sum_a_x_1 + pow((data[1][i] - mean_a_x_1), 2)
var_a_x_1 = sum_a_x_1 / (len(data[data[0] == 1]) - 1)

sum_a_x_2 = 0
for i in range(rows):
    if data[0][i] == 1:
        sum_a_x_2 = sum_a_x_2 + pow((data[2][i] - mean_a_x_2), 2)
var_a_x_2 = sum_a_x_2 / (len(data[data[0] == 1]) - 1)

sum_b_x_1 = 0
for i in range(rows):
    if data[0][i] == 0:
        sum_b_x_1 = sum_b_x_1 + pow((data[1][i] - mean_b_x_1), 2)
var_b_x_1 = sum_b_x_1 / (len(data[data[0] == 0]) - 1)

sum_b_x_2 = 0
for i in range(rows):
    if data[0][i] == 0:
        sum_b_x_2 = sum_b_x_2 + pow((data[2][i] - mean_b_x_2), 2)
var_b_x_2 = sum_b_x_2 / (len(data[data[0] == 0]) - 1)
print(var_a_x_1,var_a_x_2,var_b_x_1,var_b_x_2)
s_a_x1 = []
s_b_x1 = []

for i in range(rows):
    exponent_a_x1 = math.exp(-(math.pow(data[1][i] - mean_a_x_1, 2) / (2 * var_a_x_1)))
    p_a_x1 = exponent_a_x1 / (math.sqrt(2 * math.pi * var_a_x_1))
    s_a_x1.append(p_a_x1)
    exponent_b_x1 = math.exp(-(math.pow(data[1][i] - mean_b_x_1, 2) / (2 * var_b_x_1)))
    p_b_x1 = exponent_b_x1 / (math.sqrt(2 * math.pi * var_b_x_1))
    s_b_x1.append(p_b_x1)

s_a_x2 = []
s_b_x2 = []

for i in range(rows):
    exponent_a_x2 = math.exp(-(math.pow(data[2][i] - mean_a_x_2, 2) / (2 * var_a_x_2)))
    p_a_x2 = exponent_a_x2 / (math.sqrt(2 * math.pi * var_a_x_2))
    s_a_x2.append(p_a_x2)
    exponent_b_x2 = math.exp(-(math.pow(data[2][i] - mean_b_x_2, 2) / (2 * var_b_x_2)))
    p_b_x2 = exponent_b_x2 / (math.sqrt(2 * math.pi * var_b_x_2))
    s_b_x2.append(p_b_x2)

k_a = [a * b for a, b in zip(s_a_x1, s_a_x2)]
k_b = [a * b for a, b in zip(s_b_x1, s_b_x2)]
s = []
for i in range(rows):
    if k_a[i] > k_b[i]:
        s.append(1)
    else:
        s.append(0)
miss = 0
for i in range(rows):
    if s[i] != data[0][i]:
        miss = miss + 1

first_row = [mean_a_x_1, var_a_x_1, mean_a_x_2, var_a_x_2, p_A]
second_row = [mean_b_x_1, var_b_x_1, mean_b_x_2, var_b_x_2, p_B]
third_row = [miss]

with open(output_path, 'wt', newline='') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(first_row)
    tsv_writer.writerow(second_row)
    tsv_writer.writerow(third_row)

