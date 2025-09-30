# Authors: CommPy contributors
# License: BSD 3-Clause

"""
============================================
Utilities (:mod:`commpy.utilities`)
============================================

.. autosummary::
   :toctree: generated/

   dec2bitarray         -- Integer or array-like of integers to binary (bit array).
   decimal2bitarray     -- Specialized version for one integer to binary (bit array).
   bitarray2dec         -- Binary (bit array) to integer.
   hamming_dist         -- Hamming distance.
   euclid_dist          -- Squared Euclidean distance.
   upsample             -- Upsample by an integral factor (zero insertion).
   signal_power         -- Compute the power of a discrete time signal.
"""

from numba import njit
import numpy as np

__all__ = [
    "dec2bitarray",
    "dec2bitarray_fast",
    "bitarray2dec",
    "hamming_dist",
    "euclid_dist",
    "upsample",
    "signal_power",
]

vectorized_binary_repr = np.vectorize(np.binary_repr)


@njit(cache=True)
def dec2bitarray_fast(number: int, num_bits: int) -> np.ndarray:
    """
    Numba-compatible function to convert a decimal integer to a NumPy bit array.
    """
    result = np.zeros(num_bits, dtype=np.int8)
    for i in range(num_bits):
        result[i] = (number >> (num_bits - 1 - i)) & 1
    return result


def dec2bitarray(in_number, bit_width):
    """
    Converts a positive integer or an array-like of positive integers to NumPy array of the specified size containing
    bits (0 and 1).

    Parameters
    ----------
    in_number : int or array-like of int
        Positive integer to be converted to a bit array.

    bit_width : int
        Size of the output bit array.

    Returns
    -------
    bitarray : 1D ndarray of numpy.int8
        Array containing the binary representation of all the input decimal(s).

    """

    if isinstance(in_number, (np.integer, int)):
        return dec2bitarray_fast(in_number, bit_width)

    # Ensure the input is a NumPy array and reshape it into a column vector
    numbers = np.asarray(in_number).reshape(-1, 1)

    # Create a row vector of bit shifts, from most significant to least
    # Example for bit_width=8: [7, 6, 5, 4, 3, 2, 1, 0]
    shifts = np.arange(bit_width - 1, -1, -1)

    # Use broadcasting to apply all shifts to all numbers at once
    # 1. (numbers >> shifts) creates a 2D array of shifted values.
    # 2. (& 1) isolates the relevant bit for each element.
    # 3. .flatten() converts the 2D bit array into the final 1D array.
    bit_array = ((numbers >> shifts) & 1).astype(np.int8).flatten()

    return bit_array


@njit(cache=True)
def bitarray2dec(in_bitarray):
    """
    Converts a NumPy bit array to a decimal integer using Numba for JIT compilation.
    """
    number = 0
    for bit in in_bitarray:
        number = (number << 1) | bit
    return number


def bitarray2dec_old(in_bitarray):
    """
    Converts an input NumPy array of bits (0 and 1) to a decimal integer.

    Parameters
    ----------
    in_bitarray : 1D ndarray of ints
        Input NumPy array of bits.

    Returns
    -------
    number : int
        Integer representation of input bit array.
    """
    bit_string = "".join(map(str, in_bitarray))
    return int(bit_string, 2)


def hamming_dist(in_bitarray_1, in_bitarray_2):
    """
    Computes the Hamming distance between two NumPy arrays of bits (0 and 1).

    Parameters
    ----------
    in_bit_array_1 : 1D ndarray of ints
        NumPy array of bits.

    in_bit_array_2 : 1D ndarray of ints
        NumPy array of bits.

    Returns
    -------
    distance : int
        Hamming distance between input bit arrays.
    """

    distance = np.bitwise_xor(in_bitarray_1, in_bitarray_2).sum()

    return distance


def euclid_dist(in_array1, in_array2):
    """
    Computes the squared euclidean distance between two NumPy arrays

    Parameters
    ----------
    in_array1 : 1D ndarray of floats
        NumPy array of real values.

    in_array2 : 1D ndarray of floats
        NumPy array of real values.

    Returns
    -------
    distance : float
        Squared Euclidean distance between two input arrays.
    """
    distance = ((in_array1 - in_array2) * (in_array1 - in_array2)).sum()

    return distance


def upsample(x, n):
    """
    Upsample the input array by a factor of n

    Adds n-1 zeros between consecutive samples of x

    Parameters
    ----------
    x : 1D ndarray
        Input array.

    n : int
        Upsampling factor

    Returns
    -------
    y : 1D ndarray
        Output upsampled array.
    """
    y = np.empty(len(x) * n, dtype=complex)
    y[0::n] = x
    zero_array = np.zeros(len(x), dtype=complex)
    for i in range(1, n):
        y[i::n] = zero_array

    return y


def signal_power(signal):
    """
    Compute the power of a discrete time signal.

    Parameters
    ----------
    signal : 1D ndarray
             Input signal.

    Returns
    -------
    P : float
        Power of the input signal.
    """

    @np.vectorize
    def square_abs(s):
        return abs(s) ** 2

    P = np.mean(square_abs(signal))
    return P
