#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5 import Qt
from gnuradio import blocks, digital, gr, uhd
import sys, signal

class dual_pole_tx(gr.top_block, Qt.QWidget):
    def __init__(self):
        gr.top_block.__init__(self, "Dual-Pole TX", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Dual-Pole TX (1 MSps)")
        self.layout = Qt.QVBoxLayout(self)

        self.samp_rate, self.freq = 1e6, 2.4e9
        self.uhd_sink = uhd.usrp_sink(",".join(("", 'num_send_frames=128')), uhd.stream_args(cpu_format="fc32", args='', channels=[0,1]))
        self.uhd_sink.set_samp_rate(self.samp_rate)
        self.uhd_sink.set_time_unknown_pps(uhd.time_spec()) # Crucial for B210 Light Sync

        for i in range(2):
            self.uhd_sink.set_center_freq(self.freq, i)
            self.uhd_sink.set_gain(60, i)
            self.uhd_sink.set_antenna("TX/RX", i)

        # Modulators
        qpsk_mod = digital.generic_mod(digital.constellation_qpsk().base(), True, 4, True, 0.35)
        bpsk_mod = digital.generic_mod(digital.constellation_bpsk().base(), True, 4, True, 0.35)

        # QPSK Chain (Ch 0)
        src0 = blocks.file_source(gr.sizeof_char, '/home/sdr_caci1/Desktop/Test_Images/cameraman.jpg', True)
        ts0 = blocks.stream_to_tagged_stream(gr.sizeof_char, 1, 60, "p")
        fmt0 = digital.protocol_formatter_bb(digital.header_format_default('10010110110110100101000111011001', 0), "p")
        mux0 = blocks.tagged_stream_mux(gr.sizeof_char, "p", 0)
        self.connect(src0, ts0, (mux0, 1))
        self.connect(ts0, fmt0, (mux0, 0))
        self.connect(mux0, qpsk_mod, (self.uhd_sink, 0))

        # BPSK Chain (Ch 1)
        src1 = blocks.file_source(gr.sizeof_char, '/home/sdr_caci1/Desktop/Test_Images/caci-logo.jpg', True)
        ts1 = blocks.stream_to_tagged_stream(gr.sizeof_char, 1, 60, "p")
        fmt1 = digital.protocol_formatter_bb(digital.header_format_default('11100001010110101110100010010011', 0), "p")
        mux1 = blocks.tagged_stream_mux(gr.sizeof_char, "p", 0)
        self.connect(src1, ts1, (mux1, 1))
        self.connect(ts1, fmt1, (mux1, 0))
        self.connect(mux1, bpsk_mod, (self.uhd_sink, 1))

def main():
    qapp = Qt.QApplication(sys.argv)
    tb = dual_pole_tx(); tb.start(); tb.show()
    signal.signal(signal.SIGINT, lambda *a: Qt.QApplication.quit())
    sys.exit(qapp.exec_())

if __name__ == '__main__': main()