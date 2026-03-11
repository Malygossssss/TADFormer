#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import numpy as np
import torch
import torch.nn.functional as F


def get_output(output, task):
    output = output.permute(0, 2, 3, 1)

    if task == 'normals':
        output = (F.normalize(output, p=2, dim=3) + 1.0) * 255 / 2.0

    elif task in {'semseg', 'human_parts'}:
        _, output = torch.max(output, dim=3)

    elif task in {'edge', 'sal'}:
        output = torch.squeeze(255 * 1 / (1 + torch.exp(-output)))

    elif task in {'depth'}:
        pass

    else:
        raise ValueError('Select one of the valid tasks')

    return output


class PerformanceMeter(object):
    """ A general performance meter which shows performance across one or more tasks """

    def __init__(self, config, db_name="NYUD"):
        self.database = db_name
        self.tasks = config.TASKS
        self.meters = {t: get_single_task_meter(config,
            t, self.database) for t in self.tasks}

    def reset(self):
        for t in self.tasks:
            self.meters[t].reset()

    def update(self, pred, gt):
        for t in self.tasks:
            self.meters[t].update(pred[t], gt[t])

    def get_score(self, verbose=True):
        eval_dict = {}
        for t in self.tasks:
            eval_dict[t] = self.meters[t].get_score(verbose)

        return eval_dict

# TODO change database to handle more datasets


def get_single_task_meter(config, task, database="NYUD"):
    """ Retrieve a meter to measure the single-task performance """
    if task == 'semseg':
        from evaluation.eval_semseg import SemsegMeter
        return SemsegMeter(database, config)

    elif task == 'human_parts':
        from evaluation.eval_human_parts import HumanPartsMeter
        return HumanPartsMeter(database)

    elif task == 'normals':
        from evaluation.eval_normals import NormalsMeter
        return NormalsMeter()

    elif task == 'sal':
        from evaluation.eval_sal import SaliencyMeter
        return SaliencyMeter()

    elif task == 'depth':
        from evaluation.eval_depth import DepthMeter
        return DepthMeter()

    elif task == 'edge':
        from evaluation.eval_edge import EdgeMeter
        return EdgeMeter(pos_weight=0.95)

    else:
        raise NotImplementedError
