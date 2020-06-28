#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : action_detection.py
# Author: Chongkai LU
# Date  : 24/6/2020


def action_search(completeness_array, min_T, max_T, min_L):
    import numpy as np
    """
    Detect (temporal localization) complete action on completeness list.
    :param completeness_array: Numpy Array. List of float numbers, completeness of frames
    :param min_T: Int. Minimum completeness value threshold used to find end frame candidates.
    :param max_T: Int. Maximum completeness value threshold used to find start frame candidates.
    :param min_L: Int. Minimum complete action length used
    :return: List. List of list. each list represent a detected action illustrated as [start_inx(int) end_inx(int) loss(float)]
    Examples:
    min_T, max_T, min_L = 75, 20, 35
    """
    def is_intersect(a, b):
        if a[0] > b[1] or a[1] < b[0]:
            return False
        else:
            return True
    P = completeness_array.squeeze()
    C_startframe = np.where(P < max_T)[0]  # "C_" represent variable for candidates.
    C_endframe = np.where(P > min_T)[0]
    action_detected = []
    for s_i in C_startframe:
        for e_i in C_endframe:
            C_action_length = e_i - s_i + 1
            if C_action_length > min_L:
                action_template = np.linspace(0, 100, C_action_length)
                predicted_sequence = P[s_i:e_i + 1]
                mse = ((action_template - predicted_sequence) ** 2).mean()
                action_candidate = [s_i, e_i, mse]
                any_intersection = False
                beat_any_one = False
                for i, action in enumerate(action_detected):
                    if is_intersect(action, action_candidate):
                        any_intersection = True
                        if action_candidate[2] < action[2]:
                            beat_any_one = True
                            action_detected.pop(i)
                if beat_any_one or not any_intersection:
                    action_detected.append(action_candidate)
    action_detected.sort(key=lambda x: x[2])
    return np.array(action_detected).reshape(-1, 3)
