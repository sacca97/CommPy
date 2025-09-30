# Authors: CommPy contributors
# License: BSD 3-Clause

"""Turbo Codes"""

from numpy import array, zeros, exp, log, empty
import numpy as np
from commpy.channelcoding import conv_encode
from commpy.channelcoding.convcode import Trellis
from commpy.channelcoding.interleavers import _Interleaver
from commpy.utilities import dec2bitarray, dec2bitarray_fast
from numba import njit

# from commpy.channelcoding.map_c import backward_recursion, forward_recursion_decoding


def turbo_encode(msg_bits, trellis1, trellis2, interleaver):
    """Turbo Encoder.

    Encode Bits using a parallel concatenated rate-1/3
    turbo code consisting of two rate-1/2 systematic
    convolutional component codes.

    Parameters
    ----------
    msg_bits : 1D ndarray containing {0, 1}
        Stream of bits to be turbo encoded.

    trellis1 : Trellis object
        Trellis representation of the
        first code in the parallel concatenation.

    trellis2 : Trellis object
        Trellis representation of the
        second code in the parallel concatenation.

    interleaver : Interleaver object
        Interleaver used in the turbo code.

    Returns
    -------
    [sys_stream, non_sys_stream1, non_sys_stream2] : list of 1D ndarrays
        Encoded bit streams corresponding
        to the systematic output

        and the two non-systematic
        outputs from the two component codes.
    """

    stream = conv_encode(msg_bits, trellis1, "rsc")
    sys_stream = stream[::2]
    non_sys_stream_1 = stream[1::2]

    interlv_msg_bits = interleaver.interlv(sys_stream)
    puncture_matrix = array([[0, 1]])
    non_sys_stream_2 = conv_encode(interlv_msg_bits, trellis2, "rsc", puncture_matrix)

    sys_stream = sys_stream[0 : -trellis1.total_memory]
    non_sys_stream_1 = non_sys_stream_1[0 : -trellis1.total_memory]
    non_sys_stream_2 = non_sys_stream_2[0 : -trellis2.total_memory]

    return [sys_stream, non_sys_stream_1, non_sys_stream_2]


def _compute_branch_prob(
    code_bit_0, code_bit_1, rx_symbol_0, rx_symbol_1, noise_variance
):
    # cdef np.float64_t code_symbol_0, code_symbol_1, branch_prob, x, y

    code_symbol_0 = 2 * code_bit_0 - 1
    code_symbol_1 = 2 * code_bit_1 - 1

    x = rx_symbol_0 - code_symbol_0
    y = rx_symbol_1 - code_symbol_1

    # Normalized branch transition probability
    branch_prob = exp(-(x * x + y * y) / (2 * noise_variance))

    return branch_prob


@njit(cache=True)
def backward_recursion_log_domain(
    # Trellis attributes and pre-computed BPSK values
    number_states: int,
    number_inputs: int,
    next_state_table: np.ndarray,
    msg_bpsk: np.ndarray,
    parity_bpsk: np.ndarray,
    # Main inputs (note: priors are now log_priors)
    msg_length: int,
    noise_variance: float,
    sys_symbols: np.ndarray,
    non_sys_symbols: np.ndarray,
    log_priors: np.ndarray,
    # Output arrays (note: b_state_metrics is now log_b_state_metrics)
    log_b_state_metrics: np.ndarray,
):
    """
    A Numba-accelerated backward recursion in the log-domain.
    """
    inv_2var = -0.5 / noise_variance
    indexed_b_metrics = np.empty((number_states, number_inputs), dtype=np.float64)

    # Step 2: Backward Recursion in Log-Domain
    for reverse_time_index in range(msg_length, 0, -1):
        time_idx = reverse_time_index - 1

        # 1. Calculate branch log-probabilities (log_gamma) directly
        rx_sys = sys_symbols[time_idx]
        rx_parity = non_sys_symbols[time_idx]
        msg_contrib = (rx_sys - msg_bpsk) ** 2
        parity_contrib = (rx_parity - parity_bpsk) ** 2
        # NO exp() call here! This is the core speedup.
        log_gamma = inv_2var * (msg_contrib + parity_contrib)

        # 2. Gather next-state log-beta values
        current_log_b_metrics = log_b_state_metrics[:, reverse_time_index]
        for s in range(number_states):
            for i in range(number_inputs):
                next_state = next_state_table[s, i]
                indexed_b_metrics[s, i] = current_log_b_metrics[next_state]

        # 3. Add log-probabilities instead of multiplying probabilities
        current_log_priors = log_priors[:, time_idx]
        log_weighted_metrics = (
            log_gamma.T + current_log_priors[np.newaxis, :] + indexed_b_metrics.T
        )

        # 4. Use the LSE function instead of sum and normalize
        new_log_b_metrics = log_sum_exp_axis1(log_weighted_metrics.T)

        # 5. Store the new log-beta values. Normalization is now implicit.
        log_b_state_metrics[:, time_idx] = new_log_b_metrics


@njit(cache=True)
def backward_recursion_numba_v2(
    # Trellis attributes passed directly
    number_states: int,
    number_inputs: int,
    n: int,
    next_state_table: np.ndarray,
    output_table: np.ndarray,
    # The rest of the inputs
    msg_length: int,
    noise_variance: float,
    sys_symbols: np.ndarray,
    non_sys_symbols: np.ndarray,
    branch_probs: np.ndarray,
    priors: np.ndarray,
    b_state_metrics: np.ndarray,
):
    """
    A Numba-accelerated version of the backward recursion algorithm.
    """
    # Step 1: Pre-computation (Compiled by Numba)
    codeword_arrays = np.zeros((number_states, number_inputs, n), dtype=np.int8)
    for state in range(number_states):
        for input_val in range(number_inputs):
            code_symbol = output_table[state, input_val]
            codeword_arrays[state, input_val] = dec2bitarray_fast(code_symbol, n)

    msg_bpsk = 2 * codeword_arrays[:, :, 0] - 1
    parity_bpsk = 2 * codeword_arrays[:, :, 1] - 1
    inv_2var = -0.5 / noise_variance

    # Step 2: Backward Recursion
    for reverse_time_index in range(msg_length, 0, -1):
        time_idx = reverse_time_index - 1

        # Branch probability computation
        rx_sys = sys_symbols[time_idx]
        rx_parity = non_sys_symbols[time_idx]
        msg_contrib = (rx_sys - msg_bpsk) ** 2
        parity_contrib = (rx_parity - parity_bpsk) ** 2
        branch_prob_matrix = np.exp(inv_2var * (msg_contrib + parity_contrib))

        branch_probs[:, :, time_idx] = branch_prob_matrix.T

        # Backward metric (beta) computation
        current_priors = priors[:, time_idx]
        current_b_metrics = b_state_metrics[:, reverse_time_index]

        indexed_b_metrics = np.empty((number_states, number_inputs), dtype=np.float64)
        for s in range(number_states):
            for i in range(number_inputs):
                next_state = next_state_table[s, i]
                indexed_b_metrics[s, i] = current_b_metrics[next_state]

        weighted_metrics = (
            branch_prob_matrix * current_priors[np.newaxis, :] * indexed_b_metrics
        )

        new_b_metrics = np.sum(weighted_metrics, axis=1)

        # Normalize and update
        metric_sum = np.sum(new_b_metrics)
        if metric_sum > 0:
            b_state_metrics[:, time_idx] = new_b_metrics / metric_sum
        else:
            b_state_metrics[:, time_idx] = new_b_metrics


@njit(cache=True)
def backward_recursion_numba(
    # Trellis attributes passed directly
    number_states: int,
    number_inputs: int,
    n: int,
    next_state_table: np.ndarray,
    output_table: np.ndarray,
    # The rest of the inputs
    msg_length: int,
    noise_variance: float,
    sys_symbols: np.ndarray,
    non_sys_symbols: np.ndarray,
    branch_probs: np.ndarray,
    priors: np.ndarray,
    b_state_metrics: np.ndarray,
):
    """
    A Numba-accelerated version of the backward recursion algorithm.
    """
    # Step 1: Pre-computation (Compiled by Numba)
    codeword_arrays = np.zeros((number_states, number_inputs, n), dtype=np.int8)
    for state in range(number_states):
        for input_val in range(number_inputs):
            code_symbol = output_table[state, input_val]
            codeword_arrays[state, input_val] = dec2bitarray_fast(code_symbol, n)

    msg_bpsk = 2 * codeword_arrays[:, :, 0] - 1
    parity_bpsk = 2 * codeword_arrays[:, :, 1] - 1
    inv_2var = -0.5 / noise_variance

    indexed_b_metrics = np.empty((number_states, number_inputs), dtype=np.float64)

    # Step 2: Backward Recursion
    for reverse_time_index in range(msg_length, 0, -1):
        time_idx = reverse_time_index - 1

        # Branch probability computation
        rx_sys = sys_symbols[time_idx]
        rx_parity = non_sys_symbols[time_idx]
        msg_contrib = (rx_sys - msg_bpsk) ** 2
        parity_contrib = (rx_parity - parity_bpsk) ** 2
        branch_prob_matrix = np.exp(inv_2var * (msg_contrib + parity_contrib))

        branch_probs[:, :, time_idx] = branch_prob_matrix.T

        # Backward metric (beta) computation
        current_priors = priors[:, time_idx]
        current_b_metrics = b_state_metrics[:, reverse_time_index]

        for s in range(number_states):
            for i in range(number_inputs):
                next_state = next_state_table[s, i]
                indexed_b_metrics[s, i] = current_b_metrics[next_state]

        weighted_metrics = (
            branch_prob_matrix * current_priors[np.newaxis, :] * indexed_b_metrics
        )

        new_b_metrics = np.sum(weighted_metrics, axis=1)

        # Normalize and update
        metric_sum = np.sum(new_b_metrics)
        if metric_sum > 0:
            b_state_metrics[:, time_idx] = new_b_metrics / metric_sum
        else:
            b_state_metrics[:, time_idx] = new_b_metrics


def backward_recursion_v3(
    trellis: Trellis,
    msg_length: int,
    noise_variance: float,
    sys_symbols: np.ndarray,
    non_sys_symbols: np.ndarray,
    branch_probs: np.ndarray,
    priors: np.ndarray,
    b_state_metrics: np.ndarray,
):
    number_states = trellis.number_states
    number_inputs = trellis.number_inputs
    next_state_table = trellis.next_state_table
    output_table = trellis.output_table
    n = trellis.n

    # Pre-compute all codeword arrays once
    codeword_arrays = np.zeros((number_states, number_inputs, n), dtype=np.int8)
    for state in range(number_states):
        for input_val in range(number_inputs):
            code_symbol = output_table[state, input_val]
            codeword_arrays[state, input_val] = dec2bitarray(code_symbol, n)

    msg_bits = codeword_arrays[:, :, 0]
    parity_bits = codeword_arrays[:, :, 1]

    # Backward recursion with vectorization
    for reverse_time_index in reversed(range(1, msg_length + 1)):
        time_idx = reverse_time_index - 1

        # Vectorized branch probability computation
        rx_sys = sys_symbols[time_idx]
        rx_parity = non_sys_symbols[time_idx]

        msg_contrib = (rx_sys - (2 * msg_bits - 1)) ** 2
        parity_contrib = (rx_parity - (2 * parity_bits - 1)) ** 2
        branch_prob_matrix = np.exp(
            -0.5 / noise_variance * (msg_contrib + parity_contrib)
        )

        # Store branch probabilities
        branch_probs[:, :, time_idx] = branch_prob_matrix.T

        # Vectorized backward metric computation
        current_priors = priors[:, time_idx]
        current_b_metrics = b_state_metrics[:, reverse_time_index]

        # Use advanced indexing for efficient computation
        next_states = next_state_table  # Shape: (num_states, num_inputs)
        weighted_metrics = (
            branch_prob_matrix
            * current_priors[np.newaxis, :]
            * current_b_metrics[next_states]
        )

        # Sum over inputs to get new backward metrics
        b_state_metrics[:, time_idx] = np.sum(weighted_metrics, axis=1)

        # Normalize
        metric_sum = b_state_metrics[:, time_idx].sum()
        if metric_sum > 0:
            b_state_metrics[:, time_idx] /= metric_sum


def backward_recursion_v2(
    trellis: Trellis,
    msg_length,
    noise_variance,
    sys_symbols,
    non_sys_symbols,
    branch_probs,
    priors,
    b_state_metrics,
):
    """Fast backward recursion for MAP decoding."""
    n = trellis.n
    number_states = trellis.number_states
    number_inputs = trellis.number_inputs

    next_state_table = trellis.next_state_table
    output_table = trellis.output_table

    # Pre-compute all codeword arrays and extract msg/parity bits
    # This avoids repeated calls to dec2bitarray inside the loops
    codeword_arrays = np.zeros((number_states, number_inputs, n), dtype=np.int8)
    msg_bits = np.zeros((number_states, number_inputs), dtype=np.int8)
    parity_bits = np.zeros((number_states, number_inputs), dtype=np.int8)

    for state in range(number_states):
        for input_val in range(number_inputs):
            code_symbol = output_table[state, input_val]
            codeword_array = dec2bitarray(code_symbol, n)
            codeword_arrays[state, input_val] = codeword_array
            msg_bits[state, input_val] = codeword_array[0]
            parity_bits[state, input_val] = codeword_array[1]

    # Pre-allocate temporary array for backward metrics
    temp_b_metrics = np.zeros(number_states, dtype=np.float64)

    # Backward recursion
    for reverse_time_index in reversed(range(1, msg_length + 1)):
        time_idx = reverse_time_index - 1

        # Reset temporary metrics
        temp_b_metrics.fill(0.0)

        # Cache frequently accessed values for this time step
        rx_symbol_0 = sys_symbols[time_idx]
        rx_symbol_1 = non_sys_symbols[time_idx]
        current_priors = priors[:, time_idx]
        current_b_metrics = b_state_metrics[:, reverse_time_index]

        # Pre-compute branch probabilities for all state-input combinations
        # Vectorized computation of branch probabilities
        msg_contrib = (rx_symbol_0 - (2 * msg_bits - 1)) ** 2
        parity_contrib = (rx_symbol_1 - (2 * parity_bits - 1)) ** 2
        branch_prob_matrix = np.exp(
            -0.5 / noise_variance * (msg_contrib + parity_contrib)
        )

        # Store branch probabilities (transpose to match original indexing)
        branch_probs[:, :, time_idx] = branch_prob_matrix.T

        # Optimized backward metric computation
        for current_state in range(number_states):
            # Pre-fetch next states for this current_state
            next_states = next_state_table[current_state, :]

            for current_input in range(number_inputs):
                next_state = next_states[current_input]
                branch_prob = branch_prob_matrix[current_state, current_input]
                prior = current_priors[current_input]

                # Compute backward contribution
                temp_b_metrics[current_state] += (
                    current_b_metrics[next_state] * branch_prob * prior
                )

        # Update backward state metrics
        b_state_metrics[:, time_idx] = temp_b_metrics

        # Normalization with numerical stability
        metric_sum = temp_b_metrics.sum()
        if metric_sum > 0:
            b_state_metrics[:, time_idx] /= metric_sum


def _backward_recursion(
    trellis,
    msg_length,
    noise_variance,
    sys_symbols,
    non_sys_symbols,
    branch_probs,
    priors,
    b_state_metrics,
):
    n = trellis.n
    number_states = trellis.number_states
    number_inputs = trellis.number_inputs

    codeword_array = empty(n, "int")
    next_state_table = trellis.next_state_table
    output_table = trellis.output_table

    # Backward recursion
    for reverse_time_index in reversed(range(1, msg_length + 1)):
        for current_state in range(number_states):
            for current_input in range(number_inputs):
                next_state = next_state_table[current_state, current_input]
                code_symbol = output_table[current_state, current_input]
                codeword_array = dec2bitarray(code_symbol, n)
                parity_bit = codeword_array[1]
                msg_bit = codeword_array[0]
                rx_symbol_0 = sys_symbols[reverse_time_index - 1]
                rx_symbol_1 = non_sys_symbols[reverse_time_index - 1]
                branch_prob = _compute_branch_prob(
                    msg_bit, parity_bit, rx_symbol_0, rx_symbol_1, noise_variance
                )
                branch_probs[current_input, current_state, reverse_time_index - 1] = (
                    branch_prob
                )
                b_state_metrics[current_state, reverse_time_index - 1] += (
                    b_state_metrics[next_state, reverse_time_index]
                    * branch_prob
                    * priors[current_input, reverse_time_index - 1]
                )

        b_state_metrics[:, reverse_time_index - 1] /= b_state_metrics[
            :, reverse_time_index - 1
        ].sum()


@njit(cache=True)
def forward_recursion_decoding_numba(
    # Trellis attributes are passed directly as Numba can't handle custom classes
    number_states: int,
    next_state_table: np.ndarray,
    # The rest of the inputs are the same
    mode: str,
    msg_length: int,
    b_state_metrics: np.ndarray,
    f_state_metrics: np.ndarray,
    branch_probs: np.ndarray,
    app: np.ndarray,
    L_int: np.ndarray,
    priors: np.ndarray,
    L_ext: np.ndarray,
    decoded_bits: np.ndarray,
):
    """
    A Numba-accelerated version of the forward recursion decoding algorithm.
    """
    branch_probs_T = np.transpose(branch_probs, (1, 0, 2))
    next_state_flat = next_state_table.ravel()
    epsilon = 1e-15  # Use a small constant for stability

    # Forward Recursion
    for time_index in range(1, msg_length + 1):
        # --- Step 1: Alpha Calculation ---
        alpha_prev = f_state_metrics[:, 0]
        gamma_t = branch_probs_T[:, :, time_index - 1]
        priors_t = priors[:, time_index - 1]

        # CORRECTED: Use np.newaxis for broadcasting instead of .reshape()
        # This works correctly with non-contiguous array views in Numba.
        transition_values = (
            alpha_prev[:, np.newaxis] * gamma_t * priors_t[np.newaxis, :]
        )

        # Use Numba's supported np.bincount, which is efficient
        alpha_next = np.bincount(
            next_state_flat, weights=transition_values.ravel(), minlength=number_states
        )

        # --- Step 2: APP Calculation ---
        beta_next = b_state_metrics[:, time_index]

        # This explicit loop for mapping beta is a robust way to handle
        # advanced indexing in Numba.
        beta_next_mapped = np.empty_like(next_state_table, dtype=np.float64)
        for i in range(number_states):
            for j in range(next_state_table.shape[1]):
                beta_next_mapped[i, j] = beta_next[next_state_table[i, j]]

        app_terms = (alpha_prev[:, np.newaxis] * gamma_t) * beta_next_mapped

        # Numba requires mutation of array arguments to be explicit.
        app_sum = np.sum(app_terms, axis=0)
        app[0] = app_sum[0]
        app[1] = app_sum[1]

        if app[0] == 0:
            app[0] = epsilon
        if app[1] == 0:
            app[1] = epsilon

        # --- Step 3: LLR Calculation and Decoding ---
        lappr = L_int[time_index - 1] + log(app[1] / app[0])
        L_ext[time_index - 1] = lappr

        if mode == "decode":
            if lappr > 0:
                decoded_bits[time_index - 1] = 1
            else:
                decoded_bits[time_index - 1] = 0

        # --- Step 4: Normalization and State Update ---
        temp = np.sum(alpha_next)
        if temp == 0:
            temp = epsilon

        # In-place update of the f_state_metrics array view
        f_state_metrics[:, 0] = alpha_next / temp


def forward_recursion_decoding_v2(
    trellis: Trellis,
    mode: str,
    msg_length: int,
    b_state_metrics: np.ndarray,
    f_state_metrics: np.ndarray,
    branch_probs: np.ndarray,
    app: np.ndarray,
    L_int: np.ndarray,
    priors: np.ndarray,
    L_ext: np.ndarray,
    decoded_bits: np.ndarray,
):
    number_states = trellis.number_states
    next_state_table = trellis.next_state_table  # Shape: (num_states, num_inputs)

    # Pre-transpose branch_probs for easier broadcasting.
    # New shape: (num_states, num_inputs, msg_length)
    branch_probs_T = np.transpose(branch_probs, (1, 0, 2))

    # Get the flattened next_state table for use with bincount
    next_state_flat = next_state_table.ravel()

    # Forward Recursion
    for time_index in range(1, msg_length + 1):
        # --- Step 1: Vectorized Forward State Metric (alpha) Calculation ---

        # Get metrics and probabilities for the current time step
        alpha_prev = f_state_metrics[:, 0]  # Previous alpha values
        gamma_t = branch_probs_T[
            :, :, time_index - 1
        ]  # Transposed branch probabilities
        priors_t = priors[:, time_index - 1]  # Priors for inputs

        # Calculate transition values for all (current_state, current_input) pairs at once
        # This uses broadcasting: (num_states, 1) * (num_states, num_inputs) * (1, num_inputs)
        transition_values = (
            alpha_prev[:, np.newaxis] * gamma_t * priors_t[np.newaxis, :]
        )

        # Use np.bincount to efficiently sum transition values for each next_state.
        # This replaces the two inner for-loops.
        alpha_next = np.bincount(
            next_state_flat, weights=transition_values.ravel(), minlength=number_states
        )

        # --- Step 2: Vectorized APP Calculation ---

        beta_next = b_state_metrics[:, time_index]

        # Calculate all terms for the app sum at once using broadcasting
        # The term is: alpha_prev(s) * gamma(s, i) * beta_next(s')
        app_terms = (alpha_prev[:, np.newaxis] * gamma_t) * beta_next[next_state_table]

        # Sum over the current_state axis (axis 0) to get the final app values
        app[:] = np.sum(app_terms, axis=0)

        epsilon = np.finfo(float).eps
        if app[0] == 0:
            app[0] = epsilon
        if app[1] == 0:
            app[1] = epsilon
        # --- Step 3: LLR Calculation and Decoding (Unchanged) ---
        lappr = L_int[time_index - 1] + log(app[1] / app[0])
        L_ext[time_index - 1] = lappr

        if mode == "decode":
            decoded_bits[time_index - 1] = 1 if lappr > 0 else 0

        # --- Step 4: Normalization and State Update ---

        # Normalize the new forward state metrics in-place
        temp = np.sum(alpha_next)
        if temp == 0:
            temp = epsilon
        alpha_next /= temp

        # Update the state metrics for the next iteration
        f_state_metrics[:, 0] = alpha_next


def _forward_recursion_decoding(
    trellis,
    mode,
    msg_length,
    noise_variance,
    sys_symbols,
    non_sys_symbols,
    b_state_metrics,
    f_state_metrics,
    branch_probs,
    app,
    L_int,
    priors,
    L_ext,
    decoded_bits,
):
    n = trellis.n
    number_states = trellis.number_states
    number_inputs = trellis.number_inputs

    codeword_array = empty(n, "int")
    next_state_table = trellis.next_state_table
    output_table = trellis.output_table

    # Forward Recursion
    for time_index in range(1, msg_length + 1):
        app[:] = 0
        for current_state in range(number_states):
            for current_input in range(number_inputs):
                next_state = next_state_table[current_state, current_input]
                branch_prob = branch_probs[current_input, current_state, time_index - 1]
                # Compute the forward state metrics
                f_state_metrics[next_state, 1] += (
                    f_state_metrics[current_state, 0]
                    * branch_prob
                    * priors[current_input, time_index - 1]
                )

                # Compute APP
                app[current_input] += (
                    f_state_metrics[current_state, 0]
                    * branch_prob
                    * b_state_metrics[next_state, time_index]
                )

        lappr = L_int[time_index - 1] + log(app[1] / app[0])
        L_ext[time_index - 1] = lappr

        if mode == "decode":
            if lappr > 0:
                decoded_bits[time_index - 1] = 1
            else:
                decoded_bits[time_index - 1] = 0

        # Normalization of the forward state metrics
        f_state_metrics[:, 1] = f_state_metrics[:, 1] / f_state_metrics[:, 1].sum()

        f_state_metrics[:, 0] = f_state_metrics[:, 1]
        f_state_metrics[:, 1] = 0.0


@njit(cache=True)
def map_decode_numba(
    # System parameters
    sys_symbols: np.ndarray,
    non_sys_symbols: np.ndarray,
    noise_variance: float,
    L_int: np.ndarray,
    mode: str,
    msg_length: int,
    # Unpacked Trellis attributes
    number_states: int,
    number_inputs: int,
    n: int,
    next_state_table: np.ndarray,
    output_table: np.ndarray,
):
    """
    Core Numba-accelerated MAP decoder function.
    """
    # --- Initialization ---
    f_state_metrics = np.zeros((number_states, 2))
    f_state_metrics[0, 0] = 1.0

    b_state_metrics = np.zeros((number_states, msg_length + 1))
    b_state_metrics[:, msg_length] = 1.0 / number_states

    branch_probs = np.zeros((number_inputs, number_states, msg_length))
    app = np.zeros(number_inputs)
    decoded_bits = np.zeros(msg_length, dtype=np.int8)
    L_ext = np.zeros(msg_length)

    # --- Priors Calculation ---
    exp_L_int = np.exp(L_int)
    priors = np.empty((2, msg_length))
    priors[0, :] = 1.0 / (1.0 + exp_L_int)
    priors[1, :] = exp_L_int / (1.0 + exp_L_int)  # Numerically stabler than 1.0 - p[0]

    # --- Call the compiled recursion functions ---
    backward_recursion_numba(
        number_states,
        number_inputs,
        n,
        next_state_table,
        output_table,
        msg_length,
        noise_variance,
        sys_symbols,
        non_sys_symbols,
        branch_probs,
        priors,
        b_state_metrics,
    )

    forward_recursion_decoding_numba(
        number_states,
        next_state_table,
        mode,
        msg_length,
        b_state_metrics,
        f_state_metrics,
        branch_probs,
        app,
        L_int,
        priors,
        L_ext,
        decoded_bits,
    )

    # Numba prefers returning tuples over lists
    return (L_ext, decoded_bits)


def map_decode_v2_accelerated(
    sys_symbols: np.ndarray,
    non_sys_symbols: np.ndarray,
    trellis,  # The original Trellis object
    noise_variance: float,
    L_int: np.ndarray,
    mode: str = "decode",
):
    """
    User-facing wrapper for the Numba-accelerated MAP decoder.

    This function maintains the original API by accepting a Trellis object,
    but calls the high-performance Numba core function under the hood.
    """
    msg_length = len(sys_symbols)

    # Call the core Numba function with unpacked trellis attributes
    L_ext, decoded_bits = map_decode_numba(
        sys_symbols=sys_symbols,
        non_sys_symbols=non_sys_symbols,
        noise_variance=noise_variance,
        L_int=L_int,
        mode=mode,
        msg_length=msg_length,
        # Unpack trellis attributes here
        number_states=trellis.number_states,
        number_inputs=trellis.number_inputs,
        n=trellis.n,
        next_state_table=trellis.next_state_table,
        output_table=trellis.output_table,
    )

    # Return as a list to match the original function's output format
    return [L_ext, decoded_bits]


def map_decode_v2(
    sys_symbols: np.ndarray,
    non_sys_symbols: np.ndarray,
    trellis: Trellis,
    noise_variance: float,
    L_int: np.ndarray,
    mode: str = "decode",
):
    """Maximum a-posteriori probability (MAP) decoder.

    Decodes a stream of convolutionally encoded
    (rate 1/2) bits using the MAP algorithm.

    Parameters
    ----------
    sys_symbols : 1D ndarray
        Received symbols corresponding to
        the systematic (first output) bits in
        the codeword.

    non_sys_symbols : 1D ndarray
        Received symbols corresponding to the non-systematic
        (second output) bits in the codeword.

    trellis : Trellis object
        Trellis representation of the convolutional code.

    noise_variance : float
        Variance (power) of the AWGN channel.

    L_int : 1D ndarray
        Array representing the initial intrinsic
        information for all received
        symbols.

        Typically all zeros,
        corresponding to equal prior
        probabilities of bits 0 and 1.

    mode : str{'decode', 'compute'}, optional
        The mode in which the MAP decoder is used.
        'decode' mode returns the decoded bits

        along with the extrinsic information.
        'compute' mode returns only the
        extrinsic information.

    Returns
    -------
    [L_ext, decoded_bits] : list of two 1D ndarrays
        The first element of the list is the extrinsic information.
        The second element of the list is the decoded bits.

    """

    number_states = trellis.number_states
    number_inputs = trellis.number_inputs
    msg_length = len(sys_symbols)

    # Pre-allocate all arrays
    f_state_metrics = np.zeros([number_states, 2])
    f_state_metrics[0, 0] = 1.0

    b_state_metrics = np.zeros([number_states, msg_length + 1])
    b_state_metrics[:, msg_length] = 1.0 / number_states

    branch_probs = np.zeros([number_inputs, number_states, msg_length])
    app = np.zeros(number_inputs)
    decoded_bits = np.zeros(msg_length, dtype=np.int8)
    L_ext = np.zeros(msg_length)

    # Vectorized prior computation
    exp_L_int = np.exp(L_int)
    priors = np.empty([2, msg_length])
    priors[0, :] = 1.0 / (1.0 + exp_L_int)
    priors[1, :] = 1.0 - priors[0, :]

    backward_recursion_numba(
        trellis.number_states,
        trellis.number_inputs,
        trellis.n,
        trellis.next_state_table,
        trellis.output_table,
        msg_length,
        noise_variance,
        sys_symbols,
        non_sys_symbols,
        branch_probs,
        priors,
        b_state_metrics,
    )

    forward_recursion_decoding_numba(
        trellis.number_states,
        trellis.next_state_table,
        mode,
        msg_length,
        b_state_metrics,
        f_state_metrics,
        branch_probs,
        app,
        L_int,
        priors,
        L_ext,
        decoded_bits,
    )

    return [L_ext, decoded_bits]


def map_decode(
    sys_symbols, non_sys_symbols, trellis, noise_variance, L_int, mode="decode"
):
    """Maximum a-posteriori probability (MAP) decoder.

    Decodes a stream of convolutionally encoded
    (rate 1/2) bits using the MAP algorithm.

    Parameters
    ----------
    sys_symbols : 1D ndarray
        Received symbols corresponding to
        the systematic (first output) bits in
        the codeword.

    non_sys_symbols : 1D ndarray
        Received symbols corresponding to the non-systematic
        (second output) bits in the codeword.

    trellis : Trellis object
        Trellis representation of the convolutional code.

    noise_variance : float
        Variance (power) of the AWGN channel.

    L_int : 1D ndarray
        Array representing the initial intrinsic
        information for all received
        symbols.

        Typically all zeros,
        corresponding to equal prior
        probabilities of bits 0 and 1.

    mode : str{'decode', 'compute'}, optional
        The mode in which the MAP decoder is used.
        'decode' mode returns the decoded bits

        along with the extrinsic information.
        'compute' mode returns only the
        extrinsic information.

    Returns
    -------
    [L_ext, decoded_bits] : list of two 1D ndarrays
        The first element of the list is the extrinsic information.
        The second element of the list is the decoded bits.

    """

    k = trellis.k
    n = trellis.n
    rate = float(k) / n
    number_states = trellis.number_states
    number_inputs = trellis.number_inputs

    msg_length = len(sys_symbols)

    # Initialize forward state metrics (alpha)
    f_state_metrics = zeros([number_states, 2])
    f_state_metrics[0][0] = 1
    # print f_state_metrics

    # Initialize backward state metrics (beta)
    b_state_metrics = zeros([number_states, msg_length + 1])
    b_state_metrics[:, msg_length] = 1

    # Initialize branch transition probabilities (gamma)
    branch_probs = zeros([number_inputs, number_states, msg_length + 1])

    app = zeros(number_inputs)

    lappr = 0

    decoded_bits = zeros(msg_length, "int")
    L_ext = zeros(msg_length)

    priors = empty([2, msg_length])
    priors[0, :] = 1 / (1 + exp(L_int))
    priors[1, :] = 1 - priors[0, :]

    # Backward recursion
    _backward_recursion(
        trellis,
        msg_length,
        noise_variance,
        sys_symbols,
        non_sys_symbols,
        branch_probs,
        priors,
        b_state_metrics,
    )

    # Forward recursion
    _forward_recursion_decoding(
        trellis,
        mode,
        msg_length,
        noise_variance,
        sys_symbols,
        non_sys_symbols,
        b_state_metrics,
        f_state_metrics,
        branch_probs,
        app,
        L_int,
        priors,
        L_ext,
        decoded_bits,
    )

    return [L_ext, decoded_bits]


@njit(cache=True)
def interlv_numba(in_array: np.ndarray, p_array: np.ndarray) -> np.ndarray:
    """Numba-compatible interleaver function."""
    # Note: Assumes len(in_array) >= len(p_array)
    out_array = np.zeros_like(in_array)
    # Numba handles this type of advanced indexing efficiently
    out_array[p_array] = in_array[: len(p_array)]
    return out_array


@njit(cache=True)
def deinterlv_numba(in_array: np.ndarray, p_array: np.ndarray) -> np.ndarray:
    """Numba-compatible deinterleaver function."""
    # Numba handles this type of advanced indexing efficiently
    return in_array[p_array]


@njit(cache=True)
def turbo_decode_numba(
    # Input data
    sys_symbols: np.ndarray,
    non_sys_symbols_1: np.ndarray,
    non_sys_symbols_2: np.ndarray,
    L_int: np.ndarray,
    noise_variance: float,
    number_iterations: int,
    # Unpacked Trellis attributes
    number_states: int,
    number_inputs: int,
    n: int,
    next_state_table: np.ndarray,
    output_table: np.ndarray,
    # Unpacked Interleaver attributes
    p_array: np.ndarray,
):
    """Core Numba-accelerated Turbo decoder function."""
    msg_length = len(sys_symbols)

    # --- Pre-allocate all arrays ---
    L_int_1 = L_int.copy()
    L_ext_1 = np.zeros(msg_length)
    L_ext_2 = np.zeros(msg_length)
    L_int_2 = np.zeros(msg_length)
    L_2 = np.zeros(msg_length)
    decoded_bits = np.zeros(msg_length, dtype=np.int8)

    # A temporary tuple to receive the output from map_decode_numba
    map_output = (np.zeros(msg_length), np.zeros(msg_length, dtype=np.int8))

    # Pre-interleave systematic symbols
    sys_symbols_i = interlv_numba(sys_symbols, p_array)
    # previous_decoded_bits = np.zeros(msg_length, dtype=np.int8)

    # --- Main Turbo Decoding Loop ---
    for iteration_count in range(number_iterations):
        # --- MAP Decoder 1 ---
        # A Numba function must call the core Numba function, not the Python wrapper.
        map_output = map_decode_numba(
            sys_symbols,
            non_sys_symbols_1,
            noise_variance,
            L_int_1,
            "compute",
            msg_length,
            number_states,
            number_inputs,
            n,
            next_state_table,
            output_table,
        )
        L_ext_1 = map_output[0] - L_int_1  # Compute extrinsic information
        L_int_2 = interlv_numba(L_ext_1, p_array)

        # --- MAP Decoder 2 ---
        mode = "decode" if iteration_count == number_iterations - 1 else "compute"
        map_output = map_decode_numba(
            sys_symbols_i,
            non_sys_symbols_2,
            noise_variance,
            L_int_2,
            mode,
            msg_length,
            number_states,
            number_inputs,
            n,
            next_state_table,
            output_table,
        )
        L_2 = map_output[0]
        decoded_bits = map_output[1]
        # if iteration_count > 0 and np.array_equal(decoded_bits, previous_decoded_bits):
        # # The result is stable, we can exit early
        #     break
        L_ext_2 = L_2 - L_int_2  # Compute extrinsic info

        # Deinterleave for the next iteration's input
        if iteration_count < number_iterations - 1:
            L_int_1 = deinterlv_numba(L_ext_2, p_array)

    # Final deinterleaving of the decoded bits
    return deinterlv_numba(decoded_bits, p_array)


def turbo_decode_v2_accelerated(
    sys_symbols: np.ndarray,
    non_sys_symbols_1: np.ndarray,
    non_sys_symbols_2: np.ndarray,
    trellis: object,  # Trellis class instance
    noise_variance: float,
    number_iterations: int,
    interleaver: np.ndarray,  # Interleaver class instance
    L_int: np.ndarray = None,
):
    """
    User-facing wrapper for the fully Numba-accelerated Turbo decoder.
    Maintains the original API while calling the high-performance Numba core.
    """
    if L_int is None:
        L_int = np.zeros(len(sys_symbols))

    p_array_np = np.asarray(interleaver)

    # Call the core Numba function with all required data unpacked
    decoded_bits = turbo_decode_numba(
        sys_symbols,
        non_sys_symbols_1,
        non_sys_symbols_2,
        L_int,
        noise_variance,
        number_iterations,
        # Unpack trellis attributes
        trellis.number_states,
        trellis.number_inputs,
        trellis.n,
        trellis.next_state_table,
        trellis.output_table,
        # Unpack interleaver attributes
        p_array_np,
    )

    return decoded_bits


def turbo_decode_v2(
    sys_symbols: np.ndarray,
    non_sys_symbols_1: np.ndarray,
    non_sys_symbols_2: np.ndarray,
    trellis: Trellis,
    noise_variance: float,
    number_iterations: int,
    interleaver: _Interleaver,  # Or a subclass of _Interleaver,
    L_int: np.ndarray = None,
):
    """
    Fast Turbo Decoder with reduced memory allocations and improved data flow.
    """
    if L_int is None:
        L_int = zeros(len(sys_symbols))

    # Pre-allocate arrays to avoid repeated memory allocation
    L_int_1 = L_int.copy()  # Use copy to avoid modifying input
    L_ext_1 = zeros(len(sys_symbols))
    L_ext_2 = zeros(len(sys_symbols))
    L_int_2 = zeros(len(sys_symbols))
    L_2 = zeros(len(sys_symbols))

    # Pre-interleave systematic symbols once (avoid repeated interleaving)
    sys_symbols_i = interleaver.interlv(sys_symbols)

    # Pre-allocate decoded_bits array
    decoded_bits = zeros(len(sys_symbols), dtype=int)

    for iteration_count in range(number_iterations):
        # MAP Decoder - 1
        [L_ext_1_temp, decoded_bits_temp] = map_decode_v2(
            sys_symbols, non_sys_symbols_1, trellis, noise_variance, L_int_1, "compute"
        )

        # Copy results to pre-allocated arrays
        L_ext_1[:] = L_ext_1_temp
        decoded_bits[:] = decoded_bits_temp

        # Vectorized extrinsic information computation
        L_ext_1 -= L_int_1  # In-place subtraction

        # Use existing interleaver interface
        L_int_2[:] = interleaver.interlv(L_ext_1)

        # Determine mode for second decoder
        mode = "decode" if iteration_count == number_iterations - 1 else "compute"

        # MAP Decoder - 2
        [L_2_temp, decoded_bits_temp] = map_decode_v2(
            sys_symbols_i, non_sys_symbols_2, trellis, noise_variance, L_int_2, mode
        )

        # Copy results to pre-allocated arrays
        L_2[:] = L_2_temp
        decoded_bits[:] = decoded_bits_temp

        # Compute extrinsic information in-place
        L_ext_2[:] = L_2 - L_int_2

        # Deinterleave for next iteration (skip on last iteration)
        if iteration_count < number_iterations - 1:
            L_int_1[:] = interleaver.deinterlv(L_ext_2)

    # Final deinterleaving of decoded bits using existing interface
    decoded_bits = interleaver.deinterlv(decoded_bits)

    return decoded_bits


def turbo_decode(
    sys_symbols,
    non_sys_symbols_1,
    non_sys_symbols_2,
    trellis,
    noise_variance,
    number_iterations,
    interleaver,
    L_int=None,
):
    """Turbo Decoder.

    Decodes a stream of convolutionally encoded
    (rate 1/3) bits using the BCJR algorithm.

    Parameters
    ----------
    sys_symbols : 1D ndarray
        Received symbols corresponding to
        the systematic (first output) bits in the codeword.

    non_sys_symbols_1 : 1D ndarray
        Received symbols corresponding to
        the first parity bits in the codeword.

    non_sys_symbols_2 : 1D ndarray
        Received symbols corresponding to the
        second parity bits in the codeword.

    trellis : Trellis object
        Trellis representation of the convolutional codes
        used in the Turbo code.

    noise_variance : float
        Variance (power) of the AWGN channel.

    number_iterations : int
        Number of the iterations of the
        BCJR algorithm used in turbo decoding.

    interleaver : Interleaver object.
        Interleaver used in the turbo code.

    L_int : 1D ndarray
        Array representing the initial intrinsic
        information for all received
        symbols.

        Typically all zeros,
        corresponding to equal prior
        probabilities of bits 0 and 1.

    Returns
    -------
    decoded_bits : 1D ndarray of ints containing {0, 1}
        Decoded bit stream.

    """
    if L_int is None:
        L_int = zeros(len(sys_symbols))

    L_int_1 = L_int

    # Interleave systematic symbols for input to second decoder
    sys_symbols_i = interleaver.interlv(sys_symbols)

    for iteration_count in range(number_iterations):
        # MAP Decoder - 1
        [L_ext_1, decoded_bits] = map_decode(
            sys_symbols, non_sys_symbols_1, trellis, noise_variance, L_int_1, "compute"
        )

        L_ext_1 = L_ext_1 - L_int_1
        L_int_2 = interleaver.interlv(L_ext_1)
        if iteration_count == number_iterations - 1:
            mode = "decode"
        else:
            mode = "compute"

        # MAP Decoder - 2
        [L_2, decoded_bits] = map_decode(
            sys_symbols_i, non_sys_symbols_2, trellis, noise_variance, L_int_2, mode
        )
        L_ext_2 = L_2 - L_int_2
        L_int_1 = interleaver.deinterlv(L_ext_2)

    decoded_bits = interleaver.deinterlv(decoded_bits)

    return decoded_bits
