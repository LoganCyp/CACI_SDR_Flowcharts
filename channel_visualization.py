#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import signal
import numpy as np
import numba as nb
import sip
from PyQt5 import Qt
from gnuradio import analog, blocks, digital, gr, uhd, qtgui
from gnuradio.filter import firdes

##################################################
# 1. NUMBA-Compiled MIMO Equalizer Math (Sync Block)
##################################################
@nb.jit(nopython=True, fastmath=True, nogil=True)
def numba_fast_cma(in0, in1, out0, out1, mu, w00, w01, w10, w11):
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
        
        # Safety catch for math explosions
        if mag_sq0 > 10.0 or mag_sq1 > 10.0 or np.isnan(mag_sq0) or np.isnan(mag_sq1):
            w00, w01 = 1.0+0j, 0.0+0j
            w10, w11 = 0.0+0j, 1.0+0j
            continue
        
        err0 = y0 * (mag_sq0 - 1.0)
        err1 = y1 * (mag_sq1 - 1.0)
        
        x0_c = x0.conjugate()
        x1_c = x1.conjugate()
        
        # Weights update
        w00 -= mu * err0 * x0_c
        w01 -= mu * err0 * x1_c
        w10 -= mu * err1 * x0_c
        w11 -= mu * err1 * x1_c
        
    return w00, w01, w10, w11

class mimo_cma_2x2(gr.sync_block):
    def __init__(self, mu=1e-4):
        gr.sync_block.__init__(self,
            name="MIMO CMA XPIC",
            in_sig=[np.complex64, np.complex64],
            out_sig=[np.complex64, np.complex64])
        
        self.mu = mu
        self.w00 = 1.0 + 0j
        self.w01 = 0.0 + 0j
        self.w10 = 0.0 + 0j
        self.w11 = 1.0 + 0j

        # Warm up Numba so it compiles BEFORE the USRP streams data
        _dummy_in = np.zeros(1, dtype=np.complex64)
        _dummy_out = np.zeros(1, dtype=np.complex64)
        self.w00, self.w01, self.w10, self.w11 = numba_fast_cma(
            _dummy_in, _dummy_in,
            _dummy_out, _dummy_out,
            self.mu,
            self.w00, self.w01, self.w10, self.w11
        )

    def work(self, input_items, output_items):
        self.w00, self.w01, self.w10, self.w11 = numba_fast_cma(
            input_items[0], input_items[1],
            output_items[0], output_items[1],
            self.mu,
            self.w00, self.w01, self.w10, self.w11
        )
        return len(input_items[0])

##################################################
# 2. Main GUI and Flowgraph
##################################################
class dual_pole_rx(gr.top_block, Qt.QWidget):
    def __init__(self):
        gr.top_block.__init__(self, "Dual-Pole Rx System - Constellation Debug", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Dual-Pole SDR Receiver (Constellation View)")
        self.resize(1200, 600)

        self.main_layout = Qt.QHBoxLayout(self)

        ##################################################
        # Variables
        ##################################################
        self.sps = 4
        self.samp_rate = 1e6  
        self.freq = 2.4e9  
        self.excess_bw = 0.35 
        
        self.gain_ch0 = 30
        self.gain_ch1 = 30

        ##################################################
        # SDR Blocks & Processing
        ##################################################
        
        # USRP Source
        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join(("", 'num_recv_frames=128')),
            uhd.stream_args(cpu_format="fc32", args='', channels=list(range(0,2))),
        )
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)
        
        self.uhd_usrp_source_0.set_center_freq(self.freq, 0)
        self.uhd_usrp_source_0.set_antenna("TX/RX", 0)  
        self.uhd_usrp_source_0.set_gain(self.gain_ch0, 0)
        
        self.uhd_usrp_source_0.set_center_freq(self.freq, 1)
        self.uhd_usrp_source_0.set_antenna("TX/RX", 1)  
        self.uhd_usrp_source_0.set_gain(self.gain_ch1, 1)

        # Timing & Alignment
        self.agc_0 = analog.agc_cc(1e-4, 1.0, 1.0, 4000)
        self.agc_1 = analog.agc_cc(1e-4, 1.0, 1.0, 4000)
        
        self.fll_0 = digital.fll_band_edge_cc(self.sps, self.excess_bw, 45, 0.05)
        self.fll_1 = digital.fll_band_edge_cc(self.sps, self.excess_bw, 44, 0.0628)
        
        # MIMO Equalizer
        self.mimo_eq = mimo_cma_2x2()

        # Symbol Sync (Downsamples to 1 SPS so we can see dots instead of rings)
        rrc_taps = firdes.root_raised_cosine(32, 32, 1.0, self.excess_bw, 11*32)
        self.sync_0 = digital.pfb_clock_sync_ccf(self.sps, 0.0628, rrc_taps, 32, 16, 1.5, 2)
        self.sync_1 = digital.pfb_clock_sync_ccf(self.sps, 0.0628, rrc_taps, 32, 16, 1.5, 1)

        # Constellation Sinks
        self.const_sink_0 = qtgui.const_sink_c(1024, "", "Channel 0 (QPSK)")
        self.const_sink_1 = qtgui.const_sink_c(1024, "", "Channel 1 (BPSK)")

        ##################################################
        # GUI Setup
        ##################################################
        
        # --- Channel 0 UI ---
        self.qpsk_group = Qt.QGroupBox(f"Channel 0: Post-MIMO QPSK")
        self.qpsk_layout = Qt.QVBoxLayout()
        self.qpsk_group.setLayout(self.qpsk_layout)
        
        self.qpsk_layout.addWidget(sip.wrapinstance(self.const_sink_0.pyqwidget(), Qt.QWidget))
        
        self.qpsk_gain_label = Qt.QLabel(f"<b>RX 0 Gain:</b> {self.gain_ch0} dB")
        self.qpsk_layout.addWidget(self.qpsk_gain_label)
        self.qpsk_gain_slider = Qt.QSlider(Qt.Qt.Horizontal)
        self.qpsk_gain_slider.setRange(0, 90)
        self.qpsk_gain_slider.setValue(self.gain_ch0)
        self.qpsk_gain_slider.valueChanged.connect(self.set_ch0_gain)
        self.qpsk_layout.addWidget(self.qpsk_gain_slider)
        
        self.main_layout.addWidget(self.qpsk_group)

        # --- Channel 1 UI ---
        self.bpsk_group = Qt.QGroupBox(f"Channel 1: Post-MIMO BPSK")
        self.bpsk_layout = Qt.QVBoxLayout()
        self.bpsk_group.setLayout(self.bpsk_layout)
        
        self.bpsk_layout.addWidget(sip.wrapinstance(self.const_sink_1.pyqwidget(), Qt.QWidget))
        
        self.bpsk_gain_label = Qt.QLabel(f"<b>RX 1 Gain:</b> {self.gain_ch1} dB")
        self.bpsk_layout.addWidget(self.bpsk_gain_label)
        self.bpsk_gain_slider = Qt.QSlider(Qt.Qt.Horizontal)
        self.bpsk_gain_slider.setRange(0, 90)
        self.bpsk_gain_slider.setValue(self.gain_ch1)
        self.bpsk_gain_slider.valueChanged.connect(self.set_ch1_gain)
        self.bpsk_layout.addWidget(self.bpsk_gain_slider)
        
        self.main_layout.addWidget(self.bpsk_group)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.uhd_usrp_source_0, 0), (self.agc_0, 0))
        self.connect((self.uhd_usrp_source_0, 1), (self.agc_1, 0))
        
        self.connect((self.agc_0, 0), (self.fll_0, 0))
        self.connect((self.agc_1, 0), (self.fll_1, 0))
        
        # FLL -> MIMO
        self.connect((self.fll_0, 0), (self.mimo_eq, 0))
        self.connect((self.fll_1, 0), (self.mimo_eq, 1))

        # MIMO -> PFB Clock Syncs (Downsamples to 1 SPS)
        self.connect((self.mimo_eq, 0), (self.sync_0, 0))
        self.connect((self.mimo_eq, 1), (self.sync_1, 0))

        # Clock Syncs -> Constellation Plots
        self.connect((self.sync_0, 0), (self.const_sink_0, 0))
        self.connect((self.sync_1, 0), (self.const_sink_1, 0))

    def set_ch0_gain(self, value):
        self.gain_ch0 = value
        self.qpsk_gain_label.setText(f"<b>RX 0 Gain:</b> {self.gain_ch0} dB")
        self.uhd_usrp_source_0.set_gain(self.gain_ch0, 0)

    def set_ch1_gain(self, value):
        self.gain_ch1 = value
        self.bpsk_gain_label.setText(f"<b>RX 1 Gain:</b> {self.gain_ch1} dB")
        self.uhd_usrp_source_0.set_gain(self.gain_ch1, 1)

    def closeEvent(self, event):
        self.stop()
        self.wait()
        event.accept()

def main():
    qapp = Qt.QApplication(sys.argv)
    tb = dual_pole_rx()
    tb.start()
    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    qapp.exec_()

if __name__ == '__main__':
    main()