# Easier to use TrackEval 

<div align="center">

**Language**: English | [简体中文](README.md)

</div>

##  😆 O. Recent updates

***December 21, 2024***: Add the support of DanceTrack

***December 14, 2024***: Reorganized the entire framework and resolved the issue of not considering ignore regions in the dataset in the past, Truncation and occlusion lead to the problem of high FP (low MOTA) [This Issue]（ https://github.com/JackWoo0831/Easier_To_Use_TrackEval/issues/13#issuecomment -2540488858), as well as support for multi category reviews [This Issue]（ https://github.com/JackWoo0831/Easier_To_Use_TrackEval/issues/19 )


##  😀 I. Repo Description
Considering [TrackEval](https://github.com/JonathonLuiten/TrackEval). The library writing is slightly complex and may not be very beginner friendly, so I want to optimize it briefly to make evaluating custom datasets more convenient**
The most important issue in running code is the path problem, for which I have written two config templates to make configuring paths simpler
***Only supports 2D MOT datasets***

##  😘 II. The current dataset supports

### 1.  🗺️ MOT17
MOT17 only has annotations on the training set If you want to use MOT17 half for testing, you may need to modify some code

To use MOT17, modifications are required/ The 'data_root' parameter of configs/MOT17_train. yaml is set to your dataset path,
And modify the 'trackers_folder' parameter of 'tracker_stucture_comfig' to set it to the location where your tracking results are located


Then run:
```bash
python scripts/run_custom_dataset.py --config_path configs/MOT17_trian.yaml
```

To match the tracking results with the true values, please ensure that each line of your tracking results follows the following format:

```
<frame id>,< object id>,<top-left-x>,<top-left-y>,<w>,<h>,<confidence score>,-1,...
```

### 2.  🗺️ VisDrone

Note that VisDrone can be evaluated as a single category or multiple categories

If it is a single category review, please run it first

```bash
python dataset_tools/merge_visdrone_categories.py --data_root <your visdrone data root>
```


Combine the five valid categories specified by VisDrone into one valid category for evaluation together

Single class evaluation, run:


```bash
python scripts/run_custom_dataset.py --config_path configs/VisDrone_test_dev_merge_class.yaml
```

Multi class evaluation, run:

```bash
python scripts/run_custom_dataset.py --config_path configs/VisDrone_test_dev.yaml
```

Similarly, you also need to modify the 'data_root' and 'trackers_folder' in the YAML file to specify your ground truth and tracking results folder

For single category reviews, please ensure that each line of your tracking results follows the following format:


```
<frame id>,< object id>,<top-left-x>,<top-left-y>,<w>,<h>,<confidence score>,-1,...
```

For multi category reviews, please ensure that each line of your tracking results follows the following format:

```
<frame id>,< object id>,<top-left-x>,<top-left-y>,<w>,<h>,<confidence score>,<class_id>,...
```


❗❗ Note that in the process of multi class evaluation, the 'class_id' in your tracking results must be completely consistent with the * * marked as true value * * For example, for VisDrone, the valid category IDs are '1, 4, 5, 6, 9' (corresponding to pedestrian, car, van, truck, bus), So the class ID part of your tracking result must also correspond to '1, 4, 5, 6, 9', instead of '0, 1, 2, 3, 4' directly obtained by the detector This requires you to modify the part of the tracking code that writes the tracking results yourself

### 3.  🗺️ UAVDT

The annotation of UAVDT dataset is divided into three files` Gt.txt, gtw_hole. txt, gt_ignore. txt `. Among them, ` gt.txt ` is the main annotation, while ` gt_ignore. txt ` is the annotation of the area that should be ignored Therefore, we should merge these two files to **exclude matches within the ignored area, otherwise it will create an oversized FP** function:


```bash
python dataset_tools/parse_uavdt_annotations.py --data_root <your uavdt data root>
```

Subsequently run

```bash
python scripts/run_custom_dataset.py --config_path configs/UAVDT_test.yaml
```

Similarly, you also need to modify the 'data_root' and 'trackers_folder' in the YAML file to specify your ground truth and tracking results folder be careful, UAVDT has a total of 50 videos, of which 20 videos are from the test set, which is in UAVDT_test. yaml

Please ensure that each line of your tracking results follows the following format:

```
<frame id>,< object id>,<top-left-x>,<top-left-y>,<w>,<h>,<confidence score>,-1,...
```

### 4. 🗺️DanceTrack

The data format of ODanceTrack is almost identical to MT17, with only one category. Due to the lack of annotation in the test, only validation set evaluation is supported:

```bash
python scripts/run_custom_dataset.py --config_path configs/DanceTrack.yaml
```


##  😊 III. Evaluation of custom datasets: Explanation of Config

The evaluation configuration files for all datasets are stored in `/ Under 'configs', let's explain the information inside one by one If you need to evaluate a custom dataset, you only need to customize the 'config' according to the following explanation

###  ✅ 1. gt_structure_config

❗ This part is **important**, which is to configure the path of the ground truth of your dataset

The directory structure of a typical MOT dataset can be divided into two categories (which will be added later if there are any), **one is similar to MOT Challenge (for example: UAVDT)**

```
# ${data_root}:
#    |
#    |___${train_folder_name}(Optional)
#          |
#          |____${seq_name}
#                     |_______${image_folder}
#                                    |____${frame_name}
#                     |_______${gt_folder_name}
#                                    |____${gt_txt_name}
#    |___${test_folder_name}(Optional)
#          |
#          |____${seq_name}
#                     |_______${image_folder}
#                                    |____${frame_name}
#                     |_______${gt_folder_name}   
#                                    |____${gt_txt_name}

```
For example, if my MOT17 dataset is under `/data/datasets/`, the values are:
```
data_root: /data/datasets/MOT17
train_folder_name: train, #  Training set folder name
seq_name: MOT17-02-DPM, ..., MOT17-13-SDP,  #  Sequence name
image_folder: img1,  #  Name of the folder for storing sequence frame images
frame_name： 000001.jpg, ..., 000600.jpg, ... ,  #  Image Name
Gt_folder_name: gt # Truth value folder name
Gt_txt_name: gt.txt # The name of the truth file
test_folder_name: test
... Same as above
```


Of course, some datasets are not divided into training and testing sets, so ` train_folder_name `` Train_folder_name 'can also be omitted, for example, from the UAVDT dataset
**Another type is similar to YOLO format: (e.g. Visdrone)**


```
# ${data_root}:
#    |
#    |___${train_folder_name}(Optional)
#          |
#          |____${seq_folder_name}
#          |          |_______${seq_name}
#          |                         |____${frame_name}
#          |____${gt_folder_name}
#                     |____${gt_txt_name}
#    |___${test_folder_name}(Optional)
#          |
#          |____${seq_folder_name}
#          |          |_______${seq_name}
#          |                         |____${frame_name}
#          |____${gt_folder_name}
#                     |____${gt_txt_name}
```
For example, The directory structure of Visdrone's test set is as follows (the same applies to the training set and test set):

![alt text](assets/visdrone_dir.png "title")

In this directory structure, the truth folder and sequence folder are at the same level, rather than being contained within a sequence Therefore, for Visdrone, the values in the directory structure are as follows:

```
data_root: /data/datasets/VisDrone
train_folder_name: VisDrone2019-MOT-train, #  Training set folder name
Seq_folder_name: sequences # The name of the folder where the sequence is stored
seq_name: uav0000013_00000_v, ...,  #  Sequence name
frame_name： 0000002.jpg, ..., 00000200.jpg, ... ,  #  Image Name
Gt_folder_name: annotations # Truth value folder name
Gt_txt_name: {seq_name}. txt # The name of the truth file, which is related to the sequence name and written as a formatted string
test_folder_name: test
... Same as above
```


❗❗*** You only need to follow the following steps to set up:***

Select a directory structure template for a dataset (` configs/template1. yaml ` or ` configs/template2. yaml `), and then fill in the corresponding value in ` gt_stucture_comfig ` according to the folder name of your dataset,

> Attention` The value of gt_Loc_fata 'needs to be carefully checked, as it means finding the path to the truth value txt of a sequence This is a formatted string, and the variables inside are those in 'gt_stucture_comfig', so the names should be consistent If there are slight differences in your dataset, only this area needs to be modified


###  ✅ 2. tracker_structure_config

This section refers to the directory structure of your tracking result file Usually, you only need to modify 'trackers_folder', keeping everything else unchanged.

 For example, your tracking results are in `./result/my_tracker`, then set `trackers_folder` to `./result/my_tracker`


###  ✅ 3. OUTPUT_FOLDER

Indicate the directory where TrackEval output files are stored

###  ✅ 4. SEQ_INFO

Record the information of the sequence The format is

```
sequence name: sequence length
```

###  ✅ 5. CLASS_NAME_TO_CLASS_ID

The mapping between category names and category IDs is based on the dataset specifications If you have customized the merge category, this part also needs to be modified

###  ✅ 6. CLASSES_TO_EVAL

Which category should be evaluated All other categories will be discarded

###  ✅ 7. DISTRACTOR_CLASSES_NAMES

Other categories that may interfere with the category of interest, i.e. categories that are easily confused, will also be discarded For example, in MOT17, the category that easily interferes with pedestrians is person (representing a stationary person) If not needed, set it to empty


###  ✅ 8. AS_SINGLE_CLASS
Do you want to set all tracked results as one category The result of doing so is no longer to filter tracking results by category, but to consider them all This can sometimes be useful, for example, when switching between single/multi category reviews, you don't need to run a new tracking result

`enabled`:  true or false,  Indicate whether it is enabled

`single_class_id`:  int,  Indicate the category id you want to set

###  ✅ 9. ZERO_MARK

Some datasets, such as MOT17 and VisDrone, have a specific column in their ground truth that represents score, where 0 indicates not considering this gt and 1 indicates considering it If ZERO-MARK is enabled, the score column with a value of 0 will be discarded

`enabled`:  true or false,  Indicate whether it is enabled

###  ✅ 10. CROWD_IGNORE_REGION

Some datasets, such as MOT17 and VisDrone, have ignore region classes in their categories, which are generally areas that should be ignored (such as distant places)

`enabled`:  true or false,  Indicate whether it is enabled

`col_idx`:  int,  Which column in the true value annotation is represented (usually the category column)

`class_id`:  List[str],  What is the category ID representing the ignore region, please convert it to string format

###  ✅ 11. TRUNCATION

Indicate truncation annotation In the evaluation, targets with excessively large truncation can be ignored

`enabled`:  true or false,  Indicate whether it is enabled

`thresh`:  int or float,  Indicates a threshold, anything greater than this threshold will be discarded

###  ✅ 12. OCCLUSION

Indicate occlusion annotation Overly occluded targets can be ignored in the evaluation

`enabled`:  true or false,  Indicate whether it is enabled

`thresh`:  int or float,  Indicates a threshold, anything greater than this threshold will be discarded

###  ✅ 13. COL_IDX_MAP

In ground truth, the meaning of each column and its corresponding column index

###  ✅ 14. FRAME_START_IDX

int,  Indicates whether frame IDs in ground truth start from 0 or 1

##  😉 IV. An example

Below is a small demo to demonstrate I selected several annotation files from the MOT17 dataset as truth values and as tracking results for my model

Assuming the directory of the dataset is `/data/datasets/my_test_data `, its structure is as follows:

![alt text](assets/demo_structure.png "title")

Meanwhile, I will place the tracking results in the `./result/demo`

Then, the configuration file follows the `./configs/demo.yaml` 

Then run:

```bash
python scripts/run_custom_dataset.py --config_path configs/demo.yaml
```

Operation result:
![alt text](assets/demo_result.png "title")