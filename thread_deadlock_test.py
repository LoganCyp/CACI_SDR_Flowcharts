#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import signal
import io
import numpy as np
import pmt
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

# SAFETY CATCH: Prevents the CPU from crashing due to infinite noise search
MAX_BUF_SIZE = 500000  

class SignalProxy(QObject):
    image_received = pyqtSignal(str)

##################################################
# 1. Unstoppable Stream-Based Image Recovery
##################################################
# Converted from PDU to a pure Sync Block to prevent Thread Deadlocks
class ImageRecoveryStreamBlock(gr.sync_block):
    def __init__(self, out_jpg='recovered_latest.jpg'):
        gr.sync_block.__init__(self, name="Image Recovery Stream", 
                               in_sig=[np.uint8], out_sig=None)
        self.out_jpg = out_jpg
        self.buf = bytearray()
        self.proxy = SignalProxy()

    def work(self, input_items, output_items):
        in_bytes = input_items[0]
        
        # Eat all incoming bytes immediately
        if len(in_bytes) > 0:
            self.buf.extend(in_bytes.tobytes())

            # Flush buffer if it gets too large from static
            if len(self.buf) > MAX_BUF_SIZE:
                self.buf = self.buf[-MAX_BUF_SIZE:]

            s = self.buf.find(JPEG_START)
            if s >= 0:
                e = self.buf.find(JPEG_END, s)
                if e >= 0:
                    jpg = bytes(self.buf[s:e+2])
                    if self._valid_jpeg(jpg):
                        with open(self.out_jpg, "wb") as f:
                            f.write(jpg)
                        self.proxy.image_received.emit(self.out_jpg)
                        self.buf = self.buf[e+2:]
                    else:
                        del self.buf[:s+2]
                        
        return len(in_bytes)

    def _valid_jpeg(self, b):
        if len(b) < 100: return False
        if not HAS_PIL: return True 
        try:
            im = Image.open(io.BytesIO(b))
            im.verify()
            return True
        except:
            return False

##################################################
# 2. NUMBA-Compiled MIMO Equalizer Math
##################################################
@nb.jit(nopython=True, fastmath=True)
def numba_fast_cma(in0, in1, out0, out1, mu, w00, w01, w10, w11):
    n_samples = len(in0)
    for i in range(n_samples):
        x0, x1 = in0[i], in1[i]
        
        y0 = w00 * x0 + w01 * x1
        y1 = w10 * x0 + w11 * x1
        
        out0[i], out1[i] = y0, y1
        
        mag_sq0 = y0.real**2 + y0.imag**2
        mag_sq1 = y1.real**2 + y1.imag**2
        
        if mag_sq0 > 10.0 or mag_sq1 > 10.0 or np.isnan(mag_sq0) or np.isnan(mag_sq1):
            w00, w01 = 1.0+0j, 0.0+0j
            w10, w11 = 0.0+0j, 1.0+0j
            continue
        
        err0, err1 = y0 * (mag_sq0 - 1.0), y1 * (mag_sq1 - 1.0)
        x0_c, x1_c = np.conj(x0), np.conj(x1)
        
        w00 -= mu * err0 * x0_c
        w01 -= mu * err0 * x1_c
        w10 -= mu * err1 * x0_c
        w11 -= mu * err1 * x1_c
        
    return w00, w01, w10, w11

class mimo_cma_2x2(gr.sync_block):
    def __init__(self, mu=1e-4):
        gr.sync_block.__init__(self, name="MIMO CMA XPIC",
            in_sig=[np.complex64, np.complex64], out_sig=[np.complex64, np.complex64])
        self.mu = mu
        self.w00, self.w01, self.w10, self.w11 = 1.0+0j, 0.0+0j, 0.0+0j, 1.0+0j

    def work(self, input_items, output_items):
        self.w00, self.w01, self.w10, self.w11 = numba_fast_cma(
            input_items[0], input_items[1], output_items[0], output_items[1],
            self.mu, self.w00, self.w01, self.w10, self.w11)
        return len(input_items[0])

##################################################
# 3. Main GUI and Flowgraph
##################################################
class dual_pole_rx(gr.top_block, Qt.QWidget):
    def __init__(self):
        gr.top_block.__init__(self, "Dual-Pole Rx System", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Dual-Pole SDR Receiver (Deadlock-Free Stream)")
        self.resize(1200, 600)
        self.main_layout = Qt.QHBoxLayout(self)

        self.sps = 4
        self.samp_rate = 1e6  # Numba can handle 1 MSps effortlessly!
        self.freq = 2.4e9  
        self.excess_bw = 0.35 
        
        self.qpsk_access_code = '10010110110110100101000111011001'
        self.bpsk_access_code = '11100001010110101110100010010011'
        self.gain_ch0, self.gain_ch1 = 40, 40

        # --- QPSK UI ---
        self.qpsk_group, self.qpsk_layout = Qt.QGroupBox(f"Channel 0: QPSK ({self.freq/1e9} GHz)"), Qt.QVBoxLayout()
        self.qpsk_image_label = Qt.QLabel("Awaiting QPSK Image...")
        self.qpsk_image_label.setAlignment(Qt.Qt.AlignCenter)
        self.qpsk_image_label.setStyleSheet("font-size: 20px; color: #555; border: 2px dashed #aaa; background: #eee;")
        self.qpsk_image_label.setMinimumSize(400, 400)
        self.qpsk_gain_label = Qt.QLabel(f"<b>RX 0 Gain:</b> {self.gain_ch0} dB")
        self.qpsk_gain_slider = Qt.QSlider(Qt.Qt.Horizontal)
        self.qpsk_gain_slider.setRange(0, 90); self.qpsk_gain_slider.setValue(self.gain_ch0)
        self.qpsk_gain_slider.valueChanged.connect(self.set_ch0_gain)
        self.qpsk_layout.addWidget(self.qpsk_image_label); self.qpsk_layout.addWidget(self.qpsk_gain_label); self.qpsk_layout.addWidget(self.qpsk_gain_slider)
        self.qpsk_group.setLayout(self.qpsk_layout); self.main_layout.addWidget(self.qpsk_group)

        # --- BPSK UI ---
        self.bpsk_group, self.bpsk_layout = Qt.QGroupBox(f"Channel 1: BPSK ({self.freq/1e9} GHz)"), Qt.QVBoxLayout()
        self.bpsk_image_label = Qt.QLabel("Awaiting BPSK Image...")
        self.bpsk_image_label.setAlignment(Qt.Qt.AlignCenter)
        self.bpsk_image_label.setStyleSheet("font-size: 20px; color: #555; border: 2px dashed #aaa; background: #eee;")
        self.bpsk_image_label.setMinimumSize(400, 400)
        self.bpsk_gain_label = Qt.QLabel(f"<b>RX 1 Gain:</b> {self.gain_ch1} dB")
        self.bpsk_gain_slider = Qt.QSlider(Qt.Qt.Horizontal)
        self.bpsk_gain_slider.setRange(0, 90); self.bpsk_gain_slider.setValue(self.gain_ch1)
        self.bpsk_gain_slider.valueChanged.connect(self.set_ch1_gain)
        self.bpsk_layout.addWidget(self.bpsk_image_label); self.bpsk_layout.addWidget(self.bpsk_gain_label); self.bpsk_layout.addWidget(self.bpsk_gain_slider)
        self.bpsk_group.setLayout(self.bpsk_layout); self.main_layout.addWidget(self.bpsk_group)

        # SDR Blocks
        self.uhd_usrp_source_0 = uhd.usrp_source(",".join(("", 'num_recv_frames=128')), uhd.stream_args(cpu_format="fc32", args='', channels=list(range(0,2))))
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_source_0.set_center_freq(self.freq, 0); self.uhd_usrp_source_0.set_antenna("TX/RX", 0); self.uhd_usrp_source_0.set_gain(self.gain_ch0, 0)
        self.uhd_usrp_source_0.set_center_freq(self.freq, 1); self.uhd_usrp_source_0.set_antenna("TX/RX", 1); self.uhd_usrp_source_0.set_gain(self.gain_ch1, 1)

        self.agc_0, self.agc_1 = analog.agc_cc(1e-4, 1.0, 1.0, 4000), analog.agc_cc(1e-4, 1.0, 1.0, 4000)
        self.fll_0, self.fll_1 = digital.fll_band_edge_cc(self.sps, self.excess_bw, 45, 0.05), digital.fll_band_edge_cc(self.sps, self.excess_bw, 44, 0.0628)
        rrc = firdes.root_raised_cosine(32, 32, 1.0, self.excess_bw, 11*32)
        self.sync_0, self.sync_1 = digital.pfb_clock_sync_ccf(self.sps, 0.0628, rrc, 32, 16, 1.5, 2), digital.pfb_clock_sync_ccf(self.sps, 0.0628, rrc, 32, 16, 1.5, 1)

        self.mimo_eq = mimo_cma_2x2()
        self.costas_0, self.costas_1 = digital.costas_loop_cc(0.05, 4, False), digital.costas_loop_cc(0.05, 2, False)

        # --- QPSK Deadlock-Free Stream ---
        self.decoder_qpsk = digital.constellation_decoder_cb(digital.constellation_qpsk().base())
        self.diff_qpsk, self.unpack_qpsk = digital.diff_decoder_bb(4, digital.DIFF_DIFFERENTIAL), blocks.unpack_k_bits_bb(2)
        # 0 Threshold prevents false positives, tag_bb safely aligns bytes without requiring a strict packet length
        self.corr_qpsk = digital.correlate_access_code_tag_bb(self.qpsk_access_code, 0, "sync_tag")
        self.repack_qpsk = blocks.repack_bits_bb(1, 8, "sync_tag", True, gr.GR_MSB_FIRST)
        self.rec_qpsk = ImageRecoveryStreamBlock('qpsk_recovered.jpg')

        # --- BPSK Deadlock-Free Stream ---
        self.decoder_bpsk = digital.constellation_decoder_cb(digital.constellation_bpsk().base())
        self.diff_bpsk = digital.diff_decoder_bb(2, digital.DIFF_DIFFERENTIAL)
        self.corr_bpsk = digital.correlate_access_code_tag_bb(self.bpsk_access_code, 0, "sync_tag")
        self.repack_bpsk = blocks.repack_bits_bb(1, 8, "sync_tag", True, gr.GR_MSB_FIRST)
        self.rec_bpsk = ImageRecoveryStreamBlock('bpsk_recovered.jpg')

        self.rec_qpsk.proxy.image_received.connect(self.update_qpsk_image)
        self.rec_bpsk.proxy.image_received.connect(self.update_bpsk_image)

        # Flowgraph
        self.connect((self.uhd_usrp_source_0, 0), (self.agc_0, 0)); self.connect((self.uhd_usrp_source_0, 1), (self.agc_1, 0))
        self.connect((self.agc_0, 0), (self.fll_0, 0)); self.connect((self.agc_1, 0), (self.fll_1, 0))
        self.connect((self.fll_0, 0), (self.sync_0, 0)); self.connect((self.fll_1, 0), (self.sync_1, 0))
        self.connect((self.sync_0, 0), (self.mimo_eq, 0)); self.connect((self.sync_1, 0), (self.mimo_eq, 1))
        self.connect((self.mimo_eq, 0), (self.costas_0, 0)); self.connect((self.mimo_eq, 1), (self.costas_1, 0))

        # QPSK Chain (No CRC or PDUs to cause Deadlock)
        self.connect((self.costas_0, 0), (self.decoder_qpsk, 0)); self.connect((self.decoder_qpsk, 0), (self.diff_qpsk, 0))
        self.connect((self.diff_qpsk, 0), (self.unpack_qpsk, 0)); self.connect((self.unpack_qpsk, 0), (self.corr_qpsk, 0))
        self.connect((self.corr_qpsk, 0), (self.repack_qpsk, 0)); self.connect((self.repack_qpsk, 0), (self.rec_qpsk, 0))

        # BPSK Chain
        self.connect((self.costas_1, 0), (self.decoder_bpsk, 0)); self.connect((self.decoder_bpsk, 0), (self.diff_bpsk, 0))
        self.connect((self.diff_bpsk, 0), (self.corr_bpsk, 0)); self.connect((self.corr_bpsk, 0), (self.repack_bpsk, 0))
        self.connect((self.repack_bpsk, 0), (self.rec_bpsk, 0))

    def set_ch0_gain(self, v): self.gain_ch0 = v; self.qpsk_gain_label.setText(f"<b>RX 0 Gain:</b> {v} dB"); self.uhd_usrp_source_0.set_gain(v, 0)
    def set_ch1_gain(self, v): self.gain_ch1 = v; self.bpsk_gain_label.setText(f"<b>RX 1 Gain:</b> {v} dB"); self.uhd_usrp_source_0.set_gain(v, 1)

    def update_qpsk_image(self, path):
        pm = Qt.QPixmap(path)
        if not pm.isNull(): self.qpsk_image_label.setPixmap(pm.scaled(self.qpsk_image_label.size(), Qt.Qt.KeepAspectRatio, Qt.Qt.SmoothTransformation)); self.qpsk_image_label.setStyleSheet("border: none;")

    def update_bpsk_image(self, path):
        pm = Qt.QPixmap(path)
        if not pm.isNull(): self.bpsk_image_label.setPixmap(pm.scaled(self.bpsk_image_label.size(), Qt.Qt.KeepAspectRatio, Qt.Qt.SmoothTransformation)); self.bpsk_image_label.setStyleSheet("border: none;")

    def closeEvent(self, e): self.stop(); self.wait(); e.accept()

def main():
    qapp = Qt.QApplication(sys.argv); tb = dual_pole_rx(); tb.start(); tb.show()
    signal.signal(signal.SIGINT, lambda sig, frame: (tb.stop(), tb.wait(), Qt.QApplication.quit()))
    qapp.exec_()

if __name__ == '__main__': main()