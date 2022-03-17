# @Time : 2/22/2022 11:29 AM
# @Author : Alejandro Velasquez


# content of test_sample.py
# content of test_sample.py
def inc(x):
    if x < 0:
        return x - 1
    return x + 1

def test_answer():
    assert inc(3) == 4

def test_exam():
    assert exam(1)==1