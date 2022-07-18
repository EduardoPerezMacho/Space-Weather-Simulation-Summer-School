#!/usr/bin/env python
"""Space 477: Python: I

cosine approximation function
"""
__author__ = 'Eduardo Macho'
__email__ = 'edu_point@hotmail.com'

from math import factorial
from math import pi


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



# Will only run if this is run from command line as opposed to imported
if __name__ == '__main__':  # main code block
    print("cos(0) = ", cos_approx(0))
    print("cos(pi) = ", cos_approx(pi))
    print("cos(2*pi) = ", cos_approx(2*pi))
    print("more accurate cos(2*pi) = ", cos_approx(2*pi, accuracy=50))
