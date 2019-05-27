import argparse
from numpy import genfromtxt
from argparse import ArgumentParser

parser = ArgumentParser(description='Batch linear regression using Gradient descent')
parser.add_argument('-d', '--data', type=str, metavar='', required=True, help='Input file path')
parser.add_argument('-l', '--learningrate', type=float, metavar='', required=True, help='Learning rate')
parser.add_argument('-t', '--threshold', type=float, metavar='', required=True, help='Threshold')
args = parser.parse_args()

# read arguments to variable from argparse
file_loc = args.data
save_loc = '\\'.join(file_loc.split('\\')[0:-1])+'\\'
learning_rate = args.learningrate
threshold = args.threshold
target = 'solution_'+ file_loc.split('\\')[-1].split('.')[0]+'_eta'+str(learning_rate)+'_thres'+str(threshold)+'.csv'
print(target)