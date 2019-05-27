# imports required for the program
from numpy import genfromtxt
from argparse import ArgumentParser    # to read command line arguments
import csv

parser = ArgumentParser(description='Perceptron training')
parser.add_argument('-d', '--data', type=str, metavar='', required=True, help='Input file path')
parser.add_argument('-o', '--output', type=str, metavar='', required=True, help='output file path')
args = parser.parse_args()
file_loc = args.data
out_put_loc=args.output
csv_data = genfromtxt(file_loc, delimiter='\t')    # reading the data of tsv into array

cols = int(csv_data.shape[1])   # calculating the number of column in input data
rows = int(csv_data.size/cols)    # calculating the number of rows in the input data
cols = 3

iterations = 100
w_const_array = [[0 for x in range(cols)] for y in range(rows)]
w_anneal_array = [[0 for x in range(cols)] for y in range(rows)]
x_arr = [[0 for x in range(cols)] for y in range(rows)]
out_put = [[]]

def read_csv():
    y_arr = []
    with open(file_loc) as tsv_file:
        csv_reader = csv.reader(tsv_file, delimiter='\t')
        row_iter = 0
        for row in csv_reader:
            x_arr[row_iter][0] = 1
            col_iter = 1
            while col_iter < cols:
                x_arr[row_iter][col_iter] = float(row[col_iter])
                col_iter += 1
            if row[0] == "A":
                y_arr.append(1)
            else:
                y_arr.append(0)
            row_iter += 1
    return x_arr,y_arr


def calc_y_const_dash(x_arr,w_arr):
    product = 0
    for x,w in zip(x_arr,w_arr):
        product += x*w
    if product > 0:
        return 1
    else:
        return 0

x_arr, y_arr = read_csv()
w_const_array[0], w_anneal_array[0] = [0] * cols, [0] * cols
y_const_dash, y_anneal_dash= [0] * rows, [0] * rows
error1 = ""
error2 = ""

main_iteration = 0
while main_iteration <= 100:
    row_counter = 0
    while row_counter < rows:
        y_const_dash[row_counter] = calc_y_const_dash(x_arr[row_counter], w_const_array[main_iteration])
        y_anneal_dash[row_counter] = calc_y_const_dash(x_arr[row_counter], w_anneal_array[main_iteration])
        row_counter += 1
    error_const_counter, error_anneal_counter = 0, 0
    w_const_array[main_iteration + 1] = w_const_array[main_iteration]
    w_anneal_array[main_iteration + 1] = w_anneal_array[main_iteration]
    learning_rate = 1 / (main_iteration+1)
    row_counter = 0
    while row_counter < rows:
        if y_arr[row_counter]-y_const_dash[row_counter] != 0:
            error_const_counter += 1
        if y_arr[row_counter] - y_anneal_dash[row_counter] != 0:
            error_anneal_counter += 1
        col_counter = 0
        while col_counter < cols:
            w_const_array[main_iteration+1][col_counter] += x_arr[row_counter][col_counter] * (y_arr[row_counter]-y_const_dash[row_counter])
            w_anneal_array[main_iteration+1][col_counter] += (x_arr[row_counter][col_counter] * (y_arr[row_counter] - y_anneal_dash[row_counter]))*learning_rate
            col_counter += 1
        row_counter += 1
    out_put.append([error_anneal_counter,error_anneal_counter])
    error2 += str(error_anneal_counter) + "\t"
    error1 += str(error_const_counter) + "\t"
    main_iteration += 1
print(error1,"\n",error2)
with open(out_put_loc, "w+") as my_tsv:
    my_tsv.write(error1+"\n")
    my_tsv.write(error2)
