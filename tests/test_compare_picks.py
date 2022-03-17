# @Time : 2/22/2022 11:29 AM
# @Author : Alejandro Velasquez

import math
from src.appickcompare import number_from_filename, crossings, topic_from_variable
from src.appickcompare import agg_linear_trend


def test_number_from_filename():
    """
    Checks this simple function that extracts the number from the filename
    """
    exp = '16'
    obs = number_from_filename('fall21_real_apple_pick16_metadata')
    assert exp == obs


def test_topic_from_variable():
    """
    Checks if ROS topics are mapped accordingly
    """
    exp = 'wrench'
    obs = topic_from_variable(' torque_y')
    assert exp == obs


def test_crossings():
    """

    :return:
    """
    x = []
    y = []
    for i in range(360):
        y.append(math.sin(math.radians(i)))
        x.append(i)

    exp = [0, 180, 0, 180]
    obs = crossings(x, y)

    assert exp == obs


def test_agg_linear_trend():
    x = []
    y = []
    for i in range(360):
        y.append(math.sin(math.radians(i)))
        x.append(i)


    exp = 1
    obs = agg_linear_trend(y)

    assert exp == obs
