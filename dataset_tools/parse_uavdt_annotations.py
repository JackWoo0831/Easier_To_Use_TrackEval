"""
in UAVDT dataset, the `gt.txt` indicates the valid annotation, while the `gt_ignore.txt` indicates the ignored regions.

To avoid FP in ignored regions, we need to merge the gt.txt and gt_ignore.txt 
"""

import os 
import numpy as np 
import argparse
import subprocess

def main(args):

    TEST_SEQS = ['M0602', 'M1004', 'M1401', 'M1101', 'M1303', 'M0701', 'M0209', 'M1301', 'M0208', 'M0606', 'M1001', 'M0205', 'M1007', 'M0801', 'M0601', 'M0203', 'M0802', 'M0403', 'M1302', 'M1009']
    TEST_SEQS = ['M0101', 'M0202', 'M0203', 'M0402', 'M0601', 'M0604', 'M1002', 'M1003', 'M1008', 'M1202']
    
    GT_FILE = 'gt.txt'
    GT_IGNORE_FILE = 'gt_ignore.txt'

    GT_NEW_FILE = args.store_file_name

    DATA_ROOT = args.data_root 
    

    seqs = os.listdir(os.path.join(DATA_ROOT, 'UAV-benchmark-M'))

    for seq_name in seqs:
        if not seq_name in TEST_SEQS: continue 

        gt_file = os.path.join(DATA_ROOT, 'UAV-benchmark-M', seq_name, 'gt', GT_FILE)

        gt_ignore_file = os.path.join(DATA_ROOT, 'UAV-benchmark-M', seq_name, 'gt', GT_IGNORE_FILE)

        gt_new_file = os.path.join(DATA_ROOT, 'UAV-benchmark-M', seq_name, 'gt', GT_NEW_FILE)

        # first copy the gt file
        subprocess.run(['cp', gt_file, gt_new_file])

        # then add the ignore region info, note that the category of ignore region is set to 0
        with open(gt_new_file, 'a') as f_to:
            with open(gt_ignore_file, 'r') as f:
                
                while True:
                    line = f.readline().strip()

                    if not line: break 

                    # modify the class form -1 to 0
                    line = line[:-2] + '0'

                    f_to.write(line + '\n')

            f.close()

        f_to.close()            



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, default='/data/wujiapeng/datasets/UAVDT')
    parser.add_argument('--store_file_name', type=str, default='gt_merge.txt')

    args = parser.parse_args()

    main(args)

    # python dataset_tools/parse_uavdt_annotations.py