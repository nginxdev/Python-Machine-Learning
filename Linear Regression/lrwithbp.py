# imports required for the program
from sys import float_info    # to get max float for initial change in error
from csv import writer    # to write output to the target csv file
from numpy import genfromtxt    # to read the input from the input file
from argparse import ArgumentParser    # to read command line arguments

parser = ArgumentParser(description='Batch linear regression using Gradient descent')
parser.add_argument('-d', '--data', type=str, metavar='', required=True, help='Input file path')
parser.add_argument('-l', '--learningrate', type=float, metavar='', required=True, help='Learning rate')
parser.add_argument('-t', '--threshold', type=float, metavar='', required=True, help='Threshold')
args = parser.parse_args()

# read arguments to variable from argparse
learning_rate = args.learningrate
threshold = args.threshold
file_loc = args.data
save_loc = 'solution_'+ file_loc.split('\\')[-1].split('.')[0]+'_eta'+str(learning_rate)+'_thres'+str(threshold)+'.csv'
print('target file will be saved in current directory with name',save_loc)
csv_data = genfromtxt(file_loc, delimiter=',')    # reading the data of csv into array
cols = int(csv_data.shape[1])    # calculating the number of column in input data
rows = int(csv_data.size/cols)    # calculating the number of rows in the input data

# initialising all the required variables
out_put_arr = [[]]    # array to store the iteration, weights and sse
new_weights = [0] * cols    # array to store the new weights in each iteration
old_weights = [0] * cols    # array to store the final weights of an iteration
gradients = [0] * cols    # array to store the gradients
x_values = [0] * cols    # array to store the features
y = 0    # variable to store the actual output value

change_in_error = float_info.max    # variable to store the change in error initialised to max float value
sse = 0    # variable to store the sum square error


# function to return the predicted output based on the current weights and features
def model_fn(arr_weights, arr_x):
    y_dash = 0
    for w_val, x_val in zip(arr_weights, arr_x):
        y_dash += w_val*x_val
    return y_dash


# function to extract the features ( x values x1,x2... xn) from the input data in each row
def get_x_y_vals(csv_data, row, cols):
    x_values[0] = 1
    col_counter = 0
    while col_counter < cols-1:
        x_values[col_counter + 1] = csv_data[row, col_counter]
        col_counter += 1
    y = csv_data[row_counter, cols - 1]
    return x_values,y


# function to make copy of weights for next iteration
def copy_weights(new_weights, old_weights, cols):
    col_counter = 0
    while col_counter < cols:
        old_weights[col_counter] = new_weights[col_counter]
        col_counter += 1
    return old_weights


# function to append iteration, weights and sse to output array
def append_to_op(old_weights,sse,iteration,out_put_arr):
    out_put_arr.append([])
    out_put_arr[iteration].append(iteration)
    for weight in old_weights:
        out_put_arr[iteration].append(weight)
    out_put_arr[iteration].append(sse)
    return out_put_arr


# function to write the output array to csv file
def write_to_csv(filename, out_put_arr):
    with open(filename, "w+", newline='') as my_csv:
        csvWriter = writer(my_csv, delimiter=',')
        csvWriter.writerows(out_put_arr)


#main
iterator = 0   # variable to hold the iteration number
while change_in_error > threshold:    # the program iterates till the change in error is < threshold
    gradients = [0] * cols    # resetting the gradients array in each iteration
    sse_old = sse  # storing sse to sse_old before storing new sse, which is required to calculate the change in error
    sse = 0     # resetting the sse for new iteration

    row_counter = 0
    while row_counter < rows:
        old_weights = copy_weights(new_weights,old_weights, cols)    # making a copy of weights
        x_values,y = get_x_y_vals(csv_data, row_counter,cols)    # fetching x and y values for new iteration
        y_dash = model_fn(old_weights, x_values)    # fetching the predicted y value

        col_counter = 0
        while col_counter < cols:
            gradients[col_counter] += x_values[col_counter]*(y-y_dash) # calculating and adding the gradients
            col_counter += 1

        sse += (y-y_dash)**2    # calculating the sum square error
        row_counter += 1

    col_counter = 0
    while col_counter < cols:
        new_weights[col_counter] += gradients[col_counter] * learning_rate    # calculating the weights from gradients
        col_counter += 1
    print(iterator,old_weights,sse)
    out_put_arr = append_to_op(old_weights,sse,iterator,out_put_arr)    # appending the results to o/p array
    change_in_error = abs(sse_old-sse)    # calculating the change in error to decide the next iteration
    iterator += 1
# writing the output to csv file
write_to_csv(save_loc, out_put_arr)
print('target file is saved in current directory with name',save_loc)