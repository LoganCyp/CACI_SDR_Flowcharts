#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import signal
import io
import numpy as np
import numba as nb
from PyQt5 import Qt
from PyQt5.QtCore import pyqtSignal, QObject
from gnuradio import analog, blocks, digital, gr, uhd
from gnuradio.filter import firdes

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

JPEG_START = b"\xFF\xD8"
JPEG_END   = b"\xFF\xD9"
MAX_BUF_SIZE = 500000  

class SignalProxy(QObject):
    image_received = pyqtSignal(str)

# --- Deadlock-Free Image Extraction ---
class ImageRecoveryStreamBlock(gr.sync_block):
    def __init__(self, out_jpg='recovered_latest.jpg'):
        gr.sync_block.__init__(self, name="Image Recovery", in_sig=[np.uint8], out_sig=None)
        self.out_jpg = out_jpg
        self.buf = bytearray()
        self.proxy = SignalProxy()

    def work(self, input_items, output_items):
        in_bytes = input_items[0]
        if len(in_bytes) > 0:
            self.buf.extend(in_bytes.tobytes())
            if len(self.buf) > MAX_BUF_SIZE:
                self.buf = self.buf[-MAX_BUF_SIZE:]
            s = self.buf.find(JPEG_START)
            if s >= 0:
                e = self.buf.find(JPEG_END, s)
                if e >= 0:
                    jpg = bytes(self.buf[s:e+2])
                    with open(self.out_jpg, "wb") as f:
                        f.write(jpg)
                    self.proxy.image_received.emit(self.out_jpg)
                    self.buf = self.buf[e+2:]
        return len(in_bytes)

# --- Numba JIT MIMO Logic ---
@nb.jit(nopython=True, fastmath=True)
def numba_fast_cma(in0, in1, out0, out1, mu, w00, w01, w10, w11):
    for i in range(len(in0)):
        x0, x1 = in0[i], in1[i]
        y0 = w00 * x0 + w01 * x1
        y1 = w10 * x0 + w11 * x1
        out0[i], out1[i] = y0, y1
        mag_sq0, mag_sq1 = y0.real**2 + y0.imag**2, y1.real**2 + y1.imag**2
        if mag_sq0 > 10.0 or np.isnan(mag_sq0):
            w00, w01, w10, w11 = 1.0+0j, 0.0+0j, 0.0+0j, 1.0+0j
            continue
        err0, err1 = y0*(mag_sq0-1.0), y1*(mag_sq1-1.0)
        x0_c, x1_c = np.conj(x0), np.conj(x1)
        w00 -= mu * err0 * x0_c
        w01 -= mu * err0 * x1_c
        w10 -= mu * err1 * x0_c
        w11 -= mu * err1 * x1_c
    return w00, w01, w10, w11

class mimo_cma_2x2(gr.sync_block):
    def __init__(self, mu=1e-4):
        gr.sync_block.__init__(self, name="MIMO CMA", in_sig=[np.complex64, np.complex64], out_sig=[np.complex64, np.complex64])
        self.mu = mu
        self.w00, self.w01, self.w10, self.w11 = 1.0+0j, 0.0+0j, 0.0+0j, 1.0+0j

    def work(self, input_items, output_items):
        self.w00, self.w01, self.w10, self.w11 = numba_fast_cma(input_items[0], input_items[1], output_items[0], output_items[1], self.mu, self.w00, self.w01, self.w10, self.w11)
        return len(input_items[0])

class dual_pole_rx(gr.top_block, Qt.QWidget):
    def __init__(self):
        gr.top_block.__init__(self, "Dual-Pole RX", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Dual-Pole RX (1 MSps)")
        self.resize(1100, 500)
        self.layout = Qt.QHBoxLayout(self)

        self.samp_rate, self.freq = 1e6, 2.4e9
        self.uhd_source = uhd.usrp_source(",".join(("", 'num_recv_frames=256')), uhd.stream_args(cpu_format="fc32", args='', channels=[0,1]))
        self.uhd_source.set_samp_rate(self.samp_rate)
        for i in range(2):
            self.uhd_source.set_center_freq(self.freq, i)
            self.uhd_source.set_gain(45, i)

        self.mimo = mimo_cma_2x2()
        self.rec0, self.rec1 = ImageRecoveryStreamBlock('qpsk_rx.jpg'), ImageRecoveryStreamBlock('bpsk_rx.jpg')
        
        # UI Labels
        self.lbl0, self.lbl1 = Qt.QLabel("QPSK"), Qt.QLabel("BPSK")
        for l in [self.lbl0, self.lbl1]: self.layout.addWidget(l)
        self.rec0.proxy.image_received.connect(lambda p: self.lbl0.setPixmap(Qt.QPixmap(p).scaled(400, 400, Qt.Qt.KeepAspectRatio)))
        self.rec1.proxy.image_received.connect(lambda p: self.lbl1.setPixmap(Qt.QPixmap(p).scaled(400, 400, Qt.Qt.KeepAspectRatio)))

        # DSP Chain
        for i in range(2):
            agc = analog.agc_cc(1e-4, 1.0, 1.0, 4000)
            fll = digital.fll_band_edge_cc(4, 0.35, 44, 0.05)
            rrc = firdes.root_raised_cosine(32, 32, 1.0, 0.35, 11*32)
            sync = digital.pfb_clock_sync_ccf(4, 0.0628, rrc, 32, 16, 1.5, 2 if i==0 else 1)
            self.connect((self.uhd_source, i), agc, fll, sync, (self.mimo, i))

        # Output QPSK
        costas0 = digital.costas_loop_cc(0.05, 4)
        dec0 = digital.constellation_decoder_cb(digital.constellation_qpsk().base())
        corr0 = digital.correlate_access_code_tag_bb('10010110110110100101000111011001', 0, "s")
        self.connect((self.mimo, 0), costas0, dec0, digital.diff_decoder_bb(4, digital.DIFF_DIFFERENTIAL), blocks.unpack_k_bits_bb(2), corr0, blocks.repack_bits_bb(1,8,"s",True), self.rec0)

        # Output BPSK
        costas1 = digital.costas_loop_cc(0.05, 2)
        dec1 = digital.constellation_decoder_cb(digital.constellation_bpsk().base())
        corr1 = digital.correlate_access_code_tag_bb('11100001010110101110100010010011', 0, "s")
        self.connect((self.mimo, 1), costas1, dec1, digital.diff_decoder_bb(2, digital.DIFF_DIFFERENTIAL), corr1, blocks.repack_bits_bb(1,8,"s",True), self.rec1)

def main():
    qapp = Qt.QApplication(sys.argv)
    tb = dual_pole_rx(); tb.start(); tb.show()
    signal.signal(signal.SIGINT, lambda *a: Qt.QApplication.quit())
    qapp.exec_()

if __name__ == '__main__': main()