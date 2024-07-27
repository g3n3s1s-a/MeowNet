import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-epoch',type= int, help='sets the num of epoch',default=10)
parser.add_argument('-save',help='saves the model',action='store_true')
parser.add_argument('-load',help='loads a state dict',action='store_true')
parser.add_argument('-view_preds',help='view predictions model makes',action='store_true')
parser.add_argument('-test',type=argparse.FileType('rb'),help='add path to own image to test the mode', default=None)
parser.add_argument('-transform',help='view what transformation looks like',action='store_true')
args = parser.parse_args()
