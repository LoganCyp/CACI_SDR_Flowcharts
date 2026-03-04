import numpy as np
from gnuradio import gr
import numba as nb

# ----------------------------------------------------------------
# Numba JIT Compiled Core (Runs outside Python GIL at C-speeds)
# ----------------------------------------------------------------
@nb.jit(nopython=True, fastmath=True, nogil=True)
def fast_mimo_cma(in0, in1, out0, out1, mu, w00, w01, w10, w11):
    n_samples = len(in0)
    
    for i in range(n_samples):
        x0 = in0[i]
        x1 = in1[i]
        
        # Matrix mixing
        y0 = w00 * x0 + w01 * x1
        y1 = w10 * x0 + w11 * x1
        
        out0[i] = y0
        out1[i] = y1
        
        # Fast Magnitude Squared
        mag_sq0 = y0.real**2 + y0.imag**2
        mag_sq1 = y1.real**2 + y1.imag**2
        
        # Safety catch for math explosions (divergence)
        if mag_sq0 > 10.0 or mag_sq1 > 10.0 or np.isnan(mag_sq0) or np.isnan(mag_sq1):
            w00, w01 = 1.0+0j, 0.0+0j
            w10, w11 = 0.0+0j, 1.0+0j
            continue
        
        # Error calculation (Constant Modulus = 1.0)
        err0 = y0 * (mag_sq0 - 1.0)
        err1 = y1 * (mag_sq1 - 1.0)
        
        x0_c = x0.conjugate()
        x1_c = x1.conjugate()
        
        # Tap weight updates
        w00 -= mu * err0 * x0_c
        w01 -= mu * err0 * x1_c
        w10 -= mu * err1 * x0_c
        w11 -= mu * err1 * x1_c
        
    return w00, w01, w10, w11

# ----------------------------------------------------------------
# GNU Radio Block Wrapper
# ----------------------------------------------------------------
class mimo_cma_2x2(gr.sync_block):
    """
    2x2 MIMO XPIC Equalizer using CMA
    Expects perfectly time-aligned samples (Place AFTER FLL/Sync, or ensure 1:1 rates)
    """
    def __init__(self, mu=1e-4):
        gr.sync_block.__init__(self,
            name="MIMO CMA XPIC",
            in_sig=[np.complex64, np.complex64],
            out_sig=[np.complex64, np.complex64])
        
        self.mu = mu
        
        # Initial tap weights (Pass-through identity matrix)
        self.w00 = 1.0 + 0j
        self.w01 = 0.0 + 0j
        self.w10 = 0.0 + 0j
        self.w11 = 1.0 + 0j

        # JIT Warmup: Force Numba to compile before flowgraph starts
        _dummy_in = np.zeros(1, dtype=np.complex64)
        _dummy_out = np.zeros(1, dtype=np.complex64)
        self.w00, self.w01, self.w10, self.w11 = fast_mimo_cma(
            _dummy_in, _dummy_in,
            _dummy_out, _dummy_out,
            self.mu,
            self.w00, self.w01, self.w10, self.w11
        )

    def work(self, input_items, output_items):
        # Pass the buffers directly to the pre-compiled Numba math
        self.w00, self.w01, self.w10, self.w11 = fast_mimo_cma(
            input_items[0], input_items[1],
            output_items[0], output_items[1],
            self.mu,
            self.w00, self.w01, self.w10, self.w11
        )
        return len(input_items[0])