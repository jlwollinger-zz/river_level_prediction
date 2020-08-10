# -*- coding: utf-8 -*-
"""
Created on Thu May 28 20:31:55 2020

@author: Wollinger
"""

POINTS_TO_TRAVERSE = [150, 200, 250, 300, 400, 450, 550, 600, 650, 700]
default_measurment_values = [0.0010, 0.0013, 0.00056, 0.000704, 0.000806, 0.000543478,
                             0.004545455, -0.002631579, 0.005555556, 0.0027]

negative_measurment_values = [0.01038, 0.006, 0.008181818, 0.010384615, 0.015, 0.015,
                              0.005510204, 0.006923077, 0.010384615, 0.0027]

MODEL_LEVEL = 0.88
SKIP_VALUE = 99999

def calculateLevel(default_mask, mask):
    mask_parmeter = calculateLinesPerMask(mask)
    mask_default = calculateLinesPerMask(default_mask)
  
    result_differences = remove_standard_deviation(mask_parmeter, mask_default)

    result = calculate_probably_level(result_differences)
        
    return result

def calculateLinesPerMask(mask):
    qt_points_traverse = []
    pixel_count = 0
    for i in POINTS_TO_TRAVERSE:
        for j in range(len(mask[i])):
            if mask[i][j] == True:
                pixel_count = pixel_count + 1
            elif pixel_count is not 0:
                qt_points_traverse.append(pixel_count)
                pixel_count = 0
            else:
                pixel_count = 0
    return qt_points_traverse
            

def remove_standard_deviation(array_one, array_two):
    difference_values = []          
    if len(array_one) != len(array_two):
        if len(array_one) > len(array_two):
            array_one.pop()
        else:
            array_two.pop()
        
    
    for i in range(len(array_one)):
        difference_values.append(array_one[i] - array_two[i])

    biggest_index = 0
    biggest_value = 0
    for i in range(len(difference_values)):
        if difference_values[i] > biggest_value:
            biggest_value = difference_values[i]
            biggest_index = i
    
    difference_values[biggest_index] = SKIP_VALUE
    
    smallest_value = 0
    smallest_index = 0
    for i in range(len(difference_values)):    
        if difference_values[i] < smallest_value:
                smallest_value  = difference_values[i]
                smallest_index = i
        
    difference_values[smallest_index] = SKIP_VALUE
        
    return difference_values



def remove_standard_deviation_single(array):
    biggest_index = 0
    biggest_value = 0
    for i in range(len(array)):
        if array[i] > biggest_value:
            biggest_value = array[i]
            biggest_index = i
    
    array.pop(biggest_index - 1)
    
    smallest_value = 0
    smallest_index = 0
    for i in range(len(array)):    
        if array[i] < smallest_value:
                smallest_value  = array[i]
                smallest_index = i
        
    array.pop(smallest_index - 1)
        
    return array
        


def calculate_probably_level(differences):
    levels = []
    for i in range(len(differences)):
        if differences[i] != SKIP_VALUE:
            pixel_factor = 0
            if differences[i] > 0:
                pixel_factor = default_measurment_values[i]
            else:
                pixel_factor = negative_measurment_values[i]
            levels.append((differences[i] * pixel_factor) + MODEL_LEVEL)
    
    levels.sort()
    mediana = 0
    
    mediana = levels[int(len(levels) / 2)]
    return sum(levels) / len(levels)
    #return mediana
    

