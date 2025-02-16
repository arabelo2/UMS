import numpy as np

def gauss_c15():
    """
    Returns the 15 optimized coefficients from Wen and Breazeale.

    These coefficients simulate the wave field of a circular planar piston transducer radiating into a fluid.

    Reference:
    Wen, J.J. and M. A. Breazeale, "Computer optimization of the Gaussian beam description of an ultrasonic field," Computational
    Acoustics, Vol.2, D. Lee, A. Cakmak, R. Vichnevetsky, Eds.
    Elsevier Science Publishers, Amsterdam, 1990, pp. 181-196.

    Returns:
        tuple: (a, b) where a and b are NumPy arrays of shape (15,)
    """
    a = np.array([
        -2.9716 + 8.6187j, -3.4811 + 0.9687j, -1.3982 - 0.8128j,  0.0773 - 0.3303j,
         2.8798 + 1.6109j,  0.1259 - 0.0957j, -0.2641 - 0.6723j, 18.0190 + 7.8291j,
         0.0518 + 0.0182j, -16.9438 - 9.9384j, 0.3708 + 5.4522j, -6.6929 + 4.0722j,
        -9.3638 - 4.9998j,  1.5872 - 15.4212j, 19.0024 + 3.6850j
    ])

    b = np.array([
         4.1869 - 5.1560j,  3.8398 - 10.8004j, 3.4355 - 16.3582j, 2.4618 - 27.7134j,
         5.4699 + 28.6319j, 1.9833 - 33.2885j, 2.9335 - 22.0151j, 6.3036 + 36.7772j,
         1.3046 - 38.4650j, 6.5889 + 37.0680j, 5.5518 + 22.4255j, 5.4013 + 16.7326j,
         5.1498 + 11.1249j, 4.9665 + 5.6855j,  4.6296 + 0.3055j
    ])

    return a, b
