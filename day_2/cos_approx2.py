# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 17:02:59 2022

@author: edu_p
"""
__author__ = 'Eduardo Macho'
__email__ = 'edu_point@hotmail.com'

from math import factorial
from math import pi
import numpy as np
import argparse

def parse_args():
  """This is a function that parses one argument and number of points"""  
  # Create an argument parser:
  parser = argparse.ArgumentParser(description = \
                                   'Cos Approximation')
  # in_var: list of 1, no type defined -> string:
  parser.add_argument('in_var', type = float, \
                      help = 'Input Variables - need one!')
  # npts: scalar value, type integer, default 5:
  parser.add_argument('-npts', \
                      help = 'another scalar (default = 5)', \
                      type = int, default = 5)
  # actually parse the data now:
  args = parser.parse_args()
  return args


def cos_terms(n, x):
    """Divide the calculus in 3 parts"""
    part1 = (-1)**n
    part2 = factorial(2*n)
    part3 = x**(2*n)
    return part1/part2*part3    


def cos_approx(x, accuracy=10):
    """Calculate the cosine of a number using Taylor series
    """
    result = sum(cos_terms(n, x) for n in range(accuracy+1)) 
    return result

# ------------------------------------------------------
# My Main code:
# ------------------------------------------------------
# parse the input arguments:
args = parse_args()
# grab the variable in_var 
#   (note, this will be a list of 1 element):
in_var = args.in_var
# grab the number of points (an integer, default 5):
npts = args.npts
print('Number of points used in the approximation: ', npts)
print('The angle of approximation: ', in_var)
print()
# Will only run if this is run from command line as opposed to imported
if __name__ == '__main__':  # main code block
    print("cos(0) = ", cos_approx(0))
    print("cos(pi) = ", cos_approx(pi))
    print("cos(2*pi) = ", cos_approx(2*pi))
    print("more accurate cos(2*pi) = ", cos_approx(2*pi, accuracy=50))
    print("using args =", cos_approx(in_var, accuracy=npts))

assert abs(cos_approx(in_var, accuracy=npts) - np.cos(in_var)) < 0.0002, "not so good"

"""
Number of points for best accuracy:
    0.0: 0
    1.57: 4
    3.14: 6
    -1.57: 4
    -3.14: 6
    6.28: 11
"""