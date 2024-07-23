import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-epoch',type= int, help='sets the num of epoch',default=10)
parser.add_argument('-save',help='saves the model',action='store_true')
parser.add_argument('-load',help='loads a state dict',action='store_true')
args = parser.parse_args()
