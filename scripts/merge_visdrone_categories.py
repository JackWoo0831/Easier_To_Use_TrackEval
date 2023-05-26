"""
merge visdrone class: pedestrain, car, van, truck, bus to a single class. 
because MOTA can only calculated on One Class.
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

    DATA_ROOT = args.data_root 
    SPLIT = SPLIT_NAME_DICT[args.split]
    STORE_FILE = args.store_file_name

    VALID_CLASS_ID = [1, 4, 5, 6, 9]
    
    if osp.exists(osp.join(DATA_ROOT, STORE_FILE)):
        os.system(f'rm -r {osp.join(DATA_ROOT, STORE_FILE)}')
    else:
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

                if not int(row[6]): continue 

                cls = int(row[7])
                if cls in VALID_CLASS_ID:
                    write_line = f'{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]},{row[6]},1,{row[8]},{row[9]}\n'
                    f.write(write_line)

                else:
                    write_line = f'{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]},{row[6]},0,{row[8]},{row[9]}\n'
                    f.write(write_line)            

        f.close()


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, default='/data/wujiapeng/datasets/VisDrone2019/VisDrone2019')
    parser.add_argument('--split', type=str, default='test', )
    parser.add_argument('--store_file_name', type=str, default='merge_cls_gt')

    args = parser.parse_args()

    main(args)

    # python scripts/merge_visdrone_categories.py