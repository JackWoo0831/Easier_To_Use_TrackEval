import os
import csv
import configparser
import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_dataset import _BaseDataset
from .. import utils
from .. import _timing
from ..utils import TrackEvalException


class CustomDataset(_BaseDataset):
    """Dataset class for MOT Challenge 2D bounding box tracking"""

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = utils.get_code_path()
        default_config = {
            'GT_FOLDER': None,  # Location of GT data
            'TRACKERS_FOLDER': None,  # Trackers location
            'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
            'CLASSES_TO_EVAL': [],  # List[str]

            'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
            'PRINT_CONFIG': True,  # Whether to print current config
            'DO_PREPROC': True,  # Whether to perform preprocessing (never done for MOT15)
            'TRACKER_SUB_FOLDER': '',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
            'SEQMAP_FOLDER': None,  # Where seqmaps are found (if None, GT_FOLDER/seqmaps)
            'SEQMAP_FILE': None,  # Directly specify seqmap file (if none use seqmap_folder/benchmark-split_to_eval)
            'SEQ_INFO': None,  # If not None, directly specify sequences to eval and their number of timesteps
            'GT_LOC_FORMAT': None,  
        }
        return default_config
    
    @staticmethod
    def update_dataset_config(default_dataset_config, yaml_dataset_config):
        """Update dataset config for customized
        
        Args:
            default_dataset_config: dict 
            yaml_dataset_config: dict

        Returns:
            newconfig, dict
        """
        dir_stucture_config = yaml_dataset_config['gt_structure_config']
        tracker_structure_config = yaml_dataset_config['tracker_structure_config']

        default_dataset_config.update(yaml_dataset_config)
        # default_dataset_config.pop('dir_stucture_config')
        default_dataset_config['GT_FOLDER'] = dir_stucture_config['data_root']

        default_dataset_config['TRACKERS_FOLDER'] = tracker_structure_config['trackers_folder']
        default_dataset_config['TRACKERS_TO_EVAL'] = tracker_structure_config['trackers_to_eval']


        return default_dataset_config

    


    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()
        # Fill non-given config values with defaults
        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())

        self.gt_structure_config = self.config['gt_structure_config']
        self.tracker_structure_config = self.config['tracker_structure_config']

        if self.gt_structure_config['has_split']:
            if self.gt_structure_config['train_or_test'] == 'train':
                train_split_fol = self.gt_structure_config['train_folder_name']
            else:
                train_split_fol = self.gt_structure_config['test_folder_name']
        else: train_split_fol = ''

        self.split_name = train_split_fol

        if self.tracker_structure_config['has_split']:
            test_split_fol = self.tracker_structure_config['split_name']
        else: test_split_fol = ''


        self.gt_fol = os.path.join(self.config['GT_FOLDER'], train_split_fol)
        self.tracker_fol = os.path.join(self.config['TRACKERS_FOLDER'], test_split_fol)

        self.should_classes_combine = False
        self.use_super_categories = False
        self.data_is_zipped = self.config['INPUT_AS_ZIP']
        self.do_preproc = self.config['DO_PREPROC']

        self.output_fol = self.config['OUTPUT_FOLDER']
        if self.output_fol is None:
            self.output_fol = self.tracker_fol

        if not os.path.exists(self.output_fol):
            os.mkdir(self.output_fol)

        self.tracker_sub_fol = self.config['TRACKER_SUB_FOLDER']
        self.output_sub_fol = self.config['OUTPUT_SUB_FOLDER']

        # Get classes to eval
        self.class_list = [cls.lower() for cls in self.config['CLASSES_TO_EVAL']]
        
        self.class_name_to_class_id = self.config['CLASS_NAME_TO_CLASS_ID']

        self.valid_class_numbers = list(self.class_name_to_class_id.values())

        # Get sequences to eval and check gt files exist
        self.seq_list, self.seq_lengths = self._get_seq_info()
        if len(self.seq_list) < 1:
            raise TrackEvalException('No sequences are selected to be evaluated.')

        # Check gt files exist
        for seq in self.seq_list:
            if not self.data_is_zipped:
                curr_file = self._get_gt_location(seq)
                if not os.path.isfile(curr_file):
                    print('GT file not found ' + curr_file)
                    raise TrackEvalException('GT file not found for sequence: ' + seq)
        if self.data_is_zipped:
            curr_file = os.path.join(self.gt_fol, 'data.zip')
            if not os.path.isfile(curr_file):
                print('GT file not found ' + curr_file)
                raise TrackEvalException('GT file not found: ' + os.path.basename(curr_file))

        # Get trackers to eval
        if self.config['TRACKERS_TO_EVAL'] is None:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            self.tracker_list = self.config['TRACKERS_TO_EVAL']

        # if no tracker names, set tracker folder as tracker name itself
        if not self.tracker_structure_config['has_tracker_name']:
            self.tracker_list = [self.tracker_fol]

        if self.config['TRACKER_DISPLAY_NAMES'] is None:
            self.tracker_to_disp = dict(zip(self.tracker_list, self.tracker_list))
        elif (self.config['TRACKERS_TO_EVAL'] is not None) and (
                len(self.config['TRACKER_DISPLAY_NAMES']) == len(self.tracker_list)):
            self.tracker_to_disp = dict(zip(self.tracker_list, self.config['TRACKER_DISPLAY_NAMES']))
        else:
            raise TrackEvalException('List of tracker files and tracker display names do not match.')

        for tracker in self.tracker_list:
            if self.data_is_zipped:
                curr_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol + '.zip')
                if not os.path.isfile(curr_file):
                    print('Tracker file not found: ' + curr_file)
                    raise TrackEvalException('Tracker file not found: ' + tracker + '/' + os.path.basename(curr_file))
            else:
                for seq in self.seq_list:
                    if self.tracker_structure_config['has_tracker_name']:
                        curr_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + '.txt')
                    else:
                        curr_file = os.path.join(self.tracker_fol, self.tracker_sub_fol, seq + '.txt')
                    if not os.path.isfile(curr_file):
                        print('Tracker file not found: ' + curr_file)
                        raise TrackEvalException(
                            'Tracker file not found: ' + tracker + '/' + self.tracker_sub_fol + '/' + os.path.basename(
                                curr_file))
                    
        # set truncation filter setting
        self.filter_truncation, self.truncation_thresh = False, 1e5
        if 'TRUNCATION' in self.config.keys() and self.config['TRUNCATION']['enabled']:
            self.filter_truncation = True
            self.truncation_thresh = self.config['TRUNCATION']['thresh']
        

        # set occlusion filter setting
        self.filter_occlusion, self.occlusion_thresh = False, 1e5
        if 'OCCLUSION' in self.config.keys() and self.config['OCCLUSION']['enabled']:
            self.filter_occlusion = True 
            self.occlusion_thresh = self.config['OCCLUSION']['thresh']

        # set zero mark setting
        self.zero_mark = False 
        if 'ZERO_MARK' in self.config.keys() and self.config['ZERO_MARK']['enabled']:
            self.zero_mark = True 

        # set ignored region
        self.filter_ignored_region = False 
        if 'CROWD_IGNORE_REGION' in self.config.keys() and self.config['CROWD_IGNORE_REGION']['enabled']:
            self.filter_ignored_region = True 


        # set column index map 
        self.col_idx_map = self.config['COL_IDX_MAP']

        # set if consider the tracking results as a SINGLE class
        self.as_single_class, self.as_single_class_id = False, 0 
        if 'AS_SINGLE_CLASS' in self.config.keys() and self.config['AS_SINGLE_CLASS']['enabled']:
            self.as_single_class = True 
            self.as_single_class_id = self.config['AS_SINGLE_CLASS']['single_class_id']


    def get_display_name(self, tracker):
        return self.tracker_to_disp[tracker]
    
    def _get_gt_location_format_dict(self, seq_name, gt_txt_name):
        """get replace dict for gt location"""
        
        ret = {
            '{data_root}': self.gt_structure_config['data_root'],
            '{split_name}': self.split_name,
            '{seq_name}': seq_name, 
            '{gt_folder_name}': self.gt_structure_config['gt_folder_name'],
            '{gt_txt_name}': gt_txt_name
        }

        return ret

    
    def _get_gt_location(self, seq_name):
        """get gt location txt file path
        
        Args:
            seq_name: str, seq name
        
        Returns:
            str, path
        """

        gt_txt_name = self.gt_structure_config['gt_txt_name']
        if 'seq_name' in gt_txt_name:
            gt_txt_name = gt_txt_name.format(seq_name=seq_name)

        # replace name to value 
        replace_dict = self._get_gt_location_format_dict(seq_name, gt_txt_name)

        items = self.gt_structure_config['gt_loc_format'].split('/')
        
        ret = '' 

        for item in items:
            if item in replace_dict.keys():
                ret = os.path.join(ret, replace_dict[item])

        return ret    

    def _get_seq_info(self):
        seq_list = []
        seq_lengths = {}
        assert self.config["SEQ_INFO"] is not None

        seq_list = list(self.config["SEQ_INFO"].keys())
        seq_lengths = self.config["SEQ_INFO"]

        return seq_list, seq_lengths

    def _load_raw_file(self, tracker, seq, is_gt):
        """Load a file (gt or tracker) in the MOT Challenge 2D box format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes] : list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.
        [gt_extras] : list (for each timestep) of dicts (for each extra) of 1D NDArrays (for each det).

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes, tracker_confidences] : list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        """
        # File location
        if self.data_is_zipped:
            if is_gt:
                zip_file = os.path.join(self.gt_fol, 'data.zip')
            else:
                zip_file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol + '.zip')
            file = seq + '.txt'
        else:
            zip_file = None
            if is_gt:
                file = self._get_gt_location(seq)
            else:
                if self.tracker_structure_config['has_tracker_name']:
                    file = os.path.join(self.tracker_fol, tracker, self.tracker_sub_fol, seq + '.txt')
                else:
                    file = os.path.join(self.tracker_fol, self.tracker_sub_fol, seq + '.txt')

        # Crowd, i.e., ignored regions
        if is_gt:
            if self.filter_ignored_region:
                crowd_ignore_filter = {self.config['CROWD_IGNORE_REGION']['col_idx']: self.config['CROWD_IGNORE_REGION']['class_id']}
            else:
                crowd_ignore_filter = None
        else:
            crowd_ignore_filter = None

        # Load raw data from text file
        read_data, ignore_data = self._load_simple_text_file(file, 
                                                             time_col=self.col_idx_map['time'], 
                                                             id_col=self.col_idx_map['id'], 
                                                             remove_negative_ids=True, 
                                                             crowd_ignore_filter=crowd_ignore_filter, 
                                                             is_zipped=self.data_is_zipped, zip_file=zip_file)

        # Convert data to required format
        num_timesteps = self.seq_lengths[seq]
        data_keys = ['ids', 'classes', 'dets']
        if is_gt:
            data_keys += ['gt_crowd_ignore_regions', 'gt_extras']
        else:
            data_keys += ['tracker_confidences']
        raw_data = {key: [None] * num_timesteps for key in data_keys}

        # Check for any extra time keys
        current_time_keys = [str(t + 1) for t in range(num_timesteps)]
        extra_time_keys = [x for x in read_data.keys() if x not in current_time_keys]
        if len(extra_time_keys) > 0:
            if is_gt:
                text = 'Ground-truth'
            else:
                text = 'Tracking'
            raise TrackEvalException(
                text + ' data contains the following invalid timesteps in seq %s: ' % seq + ', '.join(
                    [str(x) + ', ' for x in extra_time_keys]))

        for t in range(num_timesteps):
            time_key = str(t + self.config['FRAME_START_IDX'])

            if time_key in read_data.keys():

                time_data = np.asarray(read_data[time_key], dtype=np.float)
                
                raw_data['dets'][t] = np.atleast_2d(time_data[:, self.col_idx_map['bbox_start']: self.col_idx_map['bbox_end'] + 1])
                raw_data['ids'][t] = np.atleast_1d(time_data[:, self.col_idx_map['id']]).astype(int)
                
                raw_data['classes'][t] = np.atleast_1d(time_data[:, self.col_idx_map['class']]).astype(int)
                

                if is_gt:
                    # add zero mark, occlusion and truncation info
                    gt_extras_dict = dict()

                    if self.zero_mark:
                        gt_extras_dict.update({'zero_marked': np.atleast_1d(time_data[:, self.col_idx_map['score']].astype(int))})

                    if self.filter_truncation:
                        gt_extras_dict.update({'truncation': np.atleast_1d(time_data[:, self.col_idx_map['truncation']].astype(int))})

                    if self.filter_occlusion:
                        gt_extras_dict.update({'occlusion': np.atleast_1d(time_data[:, self.col_idx_map['occlusion']].astype(int))})
                    
                    raw_data['gt_extras'][t] = gt_extras_dict

                else:
                    raw_data['tracker_confidences'][t] = np.atleast_1d(time_data[:, self.col_idx_map['score']])
            else:
                raw_data['dets'][t] = np.empty((0, 4))
                raw_data['ids'][t] = np.empty(0).astype(int)
                raw_data['classes'][t] = np.empty(0).astype(int)

                if is_gt:
                    gt_extras_dict = dict()

                    if self.zero_mark:
                        gt_extras_dict.update({'zero_marked': np.empty(0)})

                    if self.filter_truncation:
                        gt_extras_dict.update({'truncation': np.empty(0)})

                    if self.filter_occlusion:
                        gt_extras_dict.update({'occlusion': np.empty(0)})
                    
                    raw_data['gt_extras'][t] = gt_extras_dict
                else:
                    raw_data['tracker_confidences'][t] = np.empty(0)

            # ignored regions
            if is_gt:

                if time_key in ignore_data.keys():
                    time_ignore = np.asarray(ignore_data[time_key], dtype=np.float)
                    raw_data['gt_crowd_ignore_regions'][t] = np.atleast_2d(time_ignore[:, self.col_idx_map['bbox_start']: self.col_idx_map['bbox_end'] + 1])
                
                else:
                    raw_data['gt_crowd_ignore_regions'][t] = np.empty((0, 4))

        if is_gt:
            key_map = {'ids': 'gt_ids',
                       'classes': 'gt_classes',
                       'dets': 'gt_dets'}
        else:
            key_map = {'ids': 'tracker_ids',
                       'classes': 'tracker_classes',
                       'dets': 'tracker_dets'}
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)
        raw_data['num_timesteps'] = num_timesteps
        raw_data['seq'] = seq
        return raw_data

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        """ Preprocess data for a single sequence for a single class ready for evaluation.
        Inputs:
             - raw_data is a dict containing the data for the sequence already read in by get_raw_seq_data().
             - cls is the class to be evaluated.
        Outputs:
             - data is a dict containing all of the information that metrics need to perform evaluation.
                It contains the following fields:
                    [num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets] : integers.
                    [gt_ids, tracker_ids, tracker_confidences]: list (for each timestep) of 1D NDArrays (for each det).
                    [gt_dets, tracker_dets]: list (for each timestep) of lists of detections.
                    [similarity_scores]: list (for each timestep) of 2D NDArrays.
        Notes:
            General preprocessing (preproc) occurs in 4 steps. Some datasets may not use all of these steps.
                1) Extract only detections relevant for the class to be evaluated (including distractor detections).
                2) Match gt dets and tracker dets. Remove tracker dets that are matched to a gt det that is of a
                    distractor class, or otherwise marked as to be removed.
                3) Remove unmatched tracker dets if they fall within a crowd ignore region or don't meet a certain
                    other criteria (e.g. are too small).
                4) Remove gt dets that were only useful for preprocessing and not for actual evaluation.
            After the above preprocessing steps, this function also calculates the number of gt and tracker detections
                and unique track ids. It also relabels gt and tracker ids to be contiguous and checks that ids are
                unique within each timestep.

        """
        # Check that input data has unique ids
        self._check_unique_ids(raw_data)

        distractor_class_names = self.config['DISTRACTOR_CLASSES_NAMES']
        
        if distractor_class_names is not None:
            distractor_classes = [self.class_name_to_class_id[x] for x in distractor_class_names]
        else:
            distractor_classes = []

        cls_id = self.class_name_to_class_id[cls]

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'tracker_confidences', 'similarity_scores']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0
        for t in range(raw_data['num_timesteps']):

            # Only extract relevant dets for this class for preproc and eval (cls + distractor classes)
            gt_class_mask = np.sum([raw_data['gt_classes'][t] == c for c in [cls_id] + distractor_classes], axis=0)
            gt_class_mask = gt_class_mask.astype(np.bool)
            gt_ids = raw_data['gt_ids'][t][gt_class_mask]
            gt_dets = raw_data['gt_dets'][t][gt_class_mask]
            gt_classes = raw_data['gt_classes'][t][gt_class_mask]

            if self.zero_mark:
                gt_zero_marked = raw_data['gt_extras'][t]['zero_marked'][gt_class_mask]
            else:
                gt_zero_marked = np.ones_like(gt_classes).astype(int)

            if self.filter_occlusion:
                gt_occlusion = raw_data['gt_extras'][t]['occlusion'][gt_class_mask]
            else:
                gt_occlusion = np.zeros_like(gt_classes).astype(int)

            if self.filter_truncation:
                gt_truncation = raw_data['gt_extras'][t]['truncation'][gt_class_mask]
            else:
                gt_truncation = np.zeros_like(gt_classes).astype(int)

            # if set as single class in evaluation, consider ALL tracking results as specified SINGLE class
            
            tracker_cls_id = cls_id if not self.as_single_class else self.as_single_class_id

            tracker_class_mask = np.atleast_1d(raw_data['tracker_classes'][t] == tracker_cls_id)
            tracker_class_mask = tracker_class_mask.astype(np.bool)
            tracker_ids = raw_data['tracker_ids'][t][tracker_class_mask]
            tracker_dets = raw_data['tracker_dets'][t][tracker_class_mask]
            tracker_confidences = raw_data['tracker_confidences'][t][tracker_class_mask]
            similarity_scores = raw_data['similarity_scores'][t][gt_class_mask, :][:, tracker_class_mask]
        

            # Match tracker and gt dets (with hungarian algorithm) and remove tracker dets which match with gt dets
            # which are labeled as belonging to a distractor class.
            to_remove_matched = np.array([], np.int)
            unmatched_indices = np.arange(tracker_ids.shape[0])
            if self.do_preproc and gt_ids.shape[0] > 0 and tracker_ids.shape[0] > 0:

                matching_scores = similarity_scores.copy()
                matching_scores[matching_scores < 0.5 - np.finfo('float').eps] = 0
                match_rows, match_cols = linear_sum_assignment(-matching_scores)
                actually_matched_mask = matching_scores[match_rows, match_cols] > 0 + np.finfo('float').eps
                match_rows = match_rows[actually_matched_mask]
                match_cols = match_cols[actually_matched_mask]

                is_distractor_class = np.isin(gt_classes[match_rows], distractor_classes)
                
                is_occluded_or_truncated = np.logical_or(
                    gt_occlusion[match_rows] > self.occlusion_thresh + np.finfo('float').eps,
                    gt_truncation[match_rows] > self.truncation_thresh + np.finfo('float').eps)
                
                to_remove_matched = np.logical_or(is_distractor_class, is_occluded_or_truncated)
                to_remove_matched = match_cols[to_remove_matched]
                unmatched_indices = np.delete(unmatched_indices, match_cols, axis=0)

            # For unmatched tracker dets, also remove those that are greater than 50% within a crowd ignore region.
            if self.filter_ignored_region:
                crowd_ignore_regions = raw_data['gt_crowd_ignore_regions'][t]
                unmatched_tracker_dets = tracker_dets[unmatched_indices, :]
                intersection_with_ignore_region = self._calculate_box_ious(unmatched_tracker_dets, crowd_ignore_regions,
                                                                       box_format='xywh', do_ioa=True)
                is_within_crowd_ignore_region = np.any(intersection_with_ignore_region > 0.5 + np.finfo('float').eps, axis=1)
                to_move_unmatched = unmatched_indices[is_within_crowd_ignore_region]
                to_remove_tracker = np.concatenate([to_remove_matched, to_move_unmatched], axis=0)
            else:
                to_remove_tracker = to_remove_matched

            # Apply preprocessing to remove all unwanted tracker dets.
            data['tracker_ids'][t] = np.delete(tracker_ids, to_remove_tracker, axis=0)
            data['tracker_dets'][t] = np.delete(tracker_dets, to_remove_tracker, axis=0)
            data['tracker_confidences'][t] = np.delete(tracker_confidences, to_remove_tracker, axis=0)
            similarity_scores = np.delete(similarity_scores, to_remove_tracker, axis=1)

            # Remove gt detections marked as to remove (zero marked), and also remove gt detections not in pedestrian
            # class (not applicable for MOT15)
            if self.do_preproc:
                gt_to_keep_mask = np.equal(gt_classes, cls_id)
                if self.zero_mark:
                    gt_to_keep_mask = (gt_to_keep_mask) & \
                                      (np.not_equal(gt_zero_marked, 0))
                
                if self.filter_occlusion:
                    gt_to_keep_mask = (gt_to_keep_mask) & \
                                      (np.less_equal(gt_occlusion, self.occlusion_thresh)) 
                
                if self.filter_truncation:
                    gt_to_keep_mask = (gt_to_keep_mask) & \
                                      (np.less_equal(gt_truncation, self.truncation_thresh))
            else:
                gt_to_keep_mask = np.equal(gt_classes, cls_id)
                
            data['gt_ids'][t] = gt_ids[gt_to_keep_mask]
            data['gt_dets'][t] = gt_dets[gt_to_keep_mask, :]
            data['similarity_scores'][t] = similarity_scores[gt_to_keep_mask]

            unique_gt_ids += list(np.unique(data['gt_ids'][t]))
            unique_tracker_ids += list(np.unique(data['tracker_ids'][t]))
            num_tracker_dets += len(data['tracker_ids'][t])
            num_gt_dets += len(data['gt_ids'][t])

        # Re-label IDs such that there are no empty IDs
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['gt_ids'][t] = gt_id_map[data['gt_ids'][t]].astype(np.int)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(len(unique_tracker_ids))
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['tracker_ids'][t] = tracker_id_map[data['tracker_ids'][t]].astype(np.int)

        # Record overview statistics.
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = len(unique_tracker_ids)
        data['num_gt_ids'] = len(unique_gt_ids)
        data['num_timesteps'] = raw_data['num_timesteps']
        data['seq'] = raw_data['seq']

        # Ensure again that ids are unique per timestep after preproc.
        self._check_unique_ids(data, after_preproc=True)

        return data

    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        similarity_scores = self._calculate_box_ious(gt_dets_t, tracker_dets_t, box_format='xywh')
        return similarity_scores
