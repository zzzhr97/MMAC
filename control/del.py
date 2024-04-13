import os
import argparse

datasets = ['LC', 'CN', 'FS']

parser = argparse.ArgumentParser()

parser.add_argument('--str', type=str, default='None')
parser.add_argument('--modelonly',action='store_true')

args = parser.parse_args()

def delFile(dirPath, str):
    print(f"Delete all files containing [{str}] in [{dirPath}]")
    for root, dirs, files in os.walk(dirPath):
        for file in files:
            if str in file:
                os.remove(os.path.join(root, file))

if args.str is not None:
    for dataset in datasets:
        delFile(f'../model/{dataset}', args.str)
        if not args.modelonly:
            delFile(f'../result/{dataset}', args.str)