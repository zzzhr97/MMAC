import os
import argparse

datasets = ['LC', 'CN', 'FS']

parser = argparse.ArgumentParser()

parser.add_argument('--str', type=str, default='None')
parser.add_argument('--delmodel', type=int, default=1, choices=[0, 1])
parser.add_argument('--delresult', type=int, default=1, choices=[0, 1])
parser.add_argument('--delsearchlog', type=int, default=1, choices=[0, 1])

args = parser.parse_args()

def delFile(dirPath, str):
    print(f"Delete all files containing [{str}] in [{dirPath}]")
    for root, dirs, files in os.walk(dirPath):
        for file in files:
            if str in file:
                os.remove(os.path.join(root, file))

if args.str is not None:
    for dataset in datasets:
        if args.delmodel == 1:
            delFile(f'../model/{dataset}', args.str)
        if args.delresult == 1:
            delFile(f'../result/{dataset}', args.str)
        if args.delsearchlog == 1:
            delFile(f'../searchlog/{dataset}', args.str)