import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='LC', choices=['LC', 'CN', 'FS'])
parser.add_argument('--str', type=str, default='None')

args = parser.parse_args()

def delFile(dirPath, str):
    print(f"Delete all files containing [{str}] in [{dirPath}]")
    for root, dirs, files in os.walk(dirPath):
        for file in files:
            if str in file:
                os.remove(os.path.join(root, file))

if args.str is not None:
    delFile(f'../model/{args.dataset}', args.str)
    delFile(f'../result/{args.dataset}', args.str)