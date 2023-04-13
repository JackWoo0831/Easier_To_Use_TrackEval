"""
run 2D MOT custom dataset with config file
"""

import sys
import os
import argparse
from multiprocessing import freeze_support

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402

def main(args):
    freeze_support()

    with open(args.config_path, 'r') as f:
        yaml_dataset_config = yaml.safe_load(f)

    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_dataset_config = trackeval.datasets.CustomDataset.get_default_dataset_config()
    updated_dataset_config = trackeval.datasets.CustomDataset.update_dataset_config(default_dataset_config, 
                                                                            yaml_dataset_config)


    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}
    config = {**default_eval_config, ** updated_dataset_config, **default_metrics_config}  # Merge default configs
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in updated_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.CustomDataset(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    evaluator.evaluate(dataset_list, metrics_list)  


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/template2.yaml', help='custom config file')

    args = parser.parse_args()

    main(args)