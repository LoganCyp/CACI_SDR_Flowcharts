import numpy as np
import numba as nb
from gnuradio import gr


@nb.jit(nopython=True, fastmath=True, nogil=True, cache=True)
def numba_cma_xpic(in0, in1, out0, out1, mu, rho, w00, w01, w10, w11):
    n = in0.shape[0]

    for i in range(n):
        x0 = in0[i]
        x1 = in1[i]

        y0 = w00 * x0 + w01 * x1
        y1 = w10 * x0 + w11 * x1

        out0[i] = y0
        out1[i] = y1

        mag_sq0 = y0.real * y0.real + y0.imag * y0.imag
        mag_sq1 = y1.real * y1.real + y1.imag * y1.imag

        if mag_sq0 > 10.0 or mag_sq1 > 10.0:
            w00 = 1.0 + 0j
            w01 = 0.0 + 0j
            w10 = 0.0 + 0j
            w11 = 1.0 + 0j
            continue

        err0 = y0 * (mag_sq0 - 1.0)
        err1 = y1 * (mag_sq1 - 1.0)

        cross = y0 * y1.conjugate()
        err0 += rho * y1 * cross.conjugate()
        err1 += rho * y0 * cross

        x0c = x0.conjugate()
        x1c = x1.conjugate()

        w00 -= mu * err0 * x0c
        w01 -= mu * err0 * x1c
        w10 -= mu * err1 * x0c
        w11 -= mu * err1 * x1c

    return w00, w01, w10, w11


class mimo_xpic_2x2(gr.sync_block):

    def __init__(self, mu=1e-4, rho=0.05):
        gr.sync_block.__init__(self,
            name="MIMO XPIC 2x2",
            in_sig=[np.complex64, np.complex64],
            out_sig=[np.complex64, np.complex64])

        self.mu = mu
        self.rho = rho
        self.w00 = 1.0 + 0j
        self.w01 = 0.0 + 0j
        self.w10 = 0.0 + 0j
        self.w11 = 1.0 + 0j

    def work(self, input_items, output_items):
        self.w00, self.w01, self.w10, self.w11 = numba_cma_xpic(
            input_items[0], input_items[1],
            output_items[0], output_items[1],
            self.mu, self.rho,
            self.w00, self.w01, self.w10, self.w11
        )
        return len(input_items[0])

    def set_mu(self, mu):
        self.mu = mu

    def set_rho(self, rho):
        self.rho = rho

    def get_weights(self):
        return [[self.w00, self.w01],
                [self.w10, self.w11]]

    def reset_weights(self):
        self.w00 = 1.0 + 0j
        self.w01 = 0.0 + 0j
        self.w10 = 0.0 + 0j
        self.w11 = 1.0 + 0j