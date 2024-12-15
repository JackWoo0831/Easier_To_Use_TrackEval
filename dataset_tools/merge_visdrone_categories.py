"""
The valid class in VisDrone are pedestrain, car, van, truck, bus
This script merge the 5 class above as ONE VALID class, so that the tracking performance 
can be evaluated jointly.

The class map:

ignored_regions: 0  # ignored regions is processed separately
pedestrain, car, van, truck, bus: -1  # valid 5 class
people: 2  # people as a distractor class
bicycle, tricycle, awning-tricycle, motor, others: 3  # treat as others

"""

import os
import os.path as osp
import argparse
import numpy as np 

def main(args):
    
    SPLIT_NAME_DICT = {
        'train': 'VisDrone2019-MOT-train',
        'val': 'VisDrone2019-MOT-val',
        'test': 'VisDrone2019-MOT-test-dev',
    }

    CLASS_ID_MAP = {
        0: 0,  # ignored regions is processed separately
        1: -1,  # valid 5 class
        4: -1, 
        5: -1, 
        6: -1, 
        9: -1, 
        2: 2,  # bicycle, tricycle, awning-tricycle, motor, others, treat as others
        3: 3,
        7: 3, 
        8: 3, 
        10: 3, 
        11: 3

    }

    DATA_ROOT = args.data_root 
    SPLIT = SPLIT_NAME_DICT[args.split]
    STORE_FILE = args.store_file_name

    
    if osp.exists(osp.join(DATA_ROOT, STORE_FILE)):
        os.system(f'rm -r {osp.join(DATA_ROOT, STORE_FILE)}')

    os.makedirs(osp.join(DATA_ROOT, STORE_FILE))
    os.makedirs(osp.join(DATA_ROOT, STORE_FILE, 'annotations'))

    gt_files = os.listdir(osp.join(DATA_ROOT, SPLIT, 'annotations'))

    for gt_file in gt_files:

        # read gt file 
        anno = np.loadtxt(
            fname=osp.join(DATA_ROOT, SPLIT, 'annotations', gt_file),
            dtype=np.int32,
            delimiter=',',
        )

        to_file = osp.join(DATA_ROOT, STORE_FILE, 'annotations', gt_file)

        with open(to_file, 'w') as f:

            for row in anno:


                cls = int(row[7])
                new_cls = CLASS_ID_MAP[cls]

                write_line = f'{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]},{row[6]},{new_cls},{row[8]},{row[9]}\n'
                f.write(write_line)

        f.close()


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, default='/data/wujiapeng/datasets/VisDrone2019/VisDrone2019')
    parser.add_argument('--split', type=str, default='test', )
    parser.add_argument('--store_file_name', type=str, default='merge_cls_gt')

    args = parser.parse_args()

    main(args)

    # python dataset_tools/merge_visdrone_categories.py