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
from gnuradio import analog, blocks, digital, gr, uhd, pdu
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

##################################################
# 1. GRC-Matched JPEG Recovery Block
##################################################
class ImageRecoveryBlock(gr.basic_block):
    def __init__(self, out_jpg='recovered_latest.jpg'):
        gr.basic_block.__init__(self, name="Image Recovery", in_sig=None, out_sig=None)
        self.message_port_register_in(pmt.intern("pdus"))
        self.set_msg_handler(pmt.intern("pdus"), self._handle)
        
        self.out_jpg = out_jpg
        self.buf = bytearray()
        self.proxy = SignalProxy()

    def _handle(self, msg):
        vec = pmt.cdr(msg)
        data = bytes(pmt.u8vector_elements(vec))
        self.buf.extend(data)

        if len(self.buf) > MAX_BUF_SIZE:
            self.buf = self.buf[-MAX_BUF_SIZE:]

        s = self.buf.find(JPEG_START)
        if s < 0: return
        e = self.buf.find(JPEG_END, s)
        if e < 0: return

        jpg = bytes(self.buf[s:e+2])

        # Verification logic matched to epy_block_0_0 in GRC 
        if self._valid_jpeg(jpg):
            with open(self.out_jpg, "wb") as f:
                f.write(jpg)
            self.proxy.image_received.emit(self.out_jpg)
            self.buf = self.buf[e+2:]
        else:
            del self.buf[:s+2]

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
# 2. Main GUI and Flowgraph
##################################################
class dual_pole_rx(gr.top_block, Qt.QWidget):
    def __init__(self):
        gr.top_block.__init__(self, "Dual-Pole Rx System", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Dual-Pole RX (GRC Equalizer Integrated)")
        self.resize(1200, 600)
        self.main_layout = Qt.QHBoxLayout(self)

        # Variables matched to GRC [cite: 52, 53]
        self.sps = 4
        self.samp_rate = 1e6 
        self.freq = 2.4e9  
        self.excess_bw = 0.35 
        
        # Access codes 
        self.qpsk_access_code = '11100001010110101110100010010011'
        self.bpsk_access_code = '11100001010110101110100010010011'

        # --- GUI Layout ---
        self.qpsk_lbl = Qt.QLabel("Awaiting QPSK..."); self.layout_q = Qt.QVBoxLayout()
        self.layout_q.addWidget(Qt.QLabel("<b>Channel 0: QPSK (Equalized)</b>")); self.layout_q.addWidget(self.qpsk_lbl)
        self.main_layout.addLayout(self.layout_q)

        self.bpsk_lbl = Qt.QLabel("Awaiting BPSK..."); self.layout_b = Qt.QVBoxLayout()
        self.layout_b.addWidget(Qt.QLabel("<b>Channel 1: BPSK (Standard)</b>")); self.layout_b.addWidget(self.bpsk_lbl)
        self.main_layout.addLayout(self.layout_b)

        # SDR Source [cite: 97, 100]
        self.uhd_source = uhd.usrp_source(",".join(("", 'num_recv_frames=256')),
            uhd.stream_args(cpu_format="fc32", args='', channels=[0,1]))
        self.uhd_source.set_samp_rate(self.samp_rate)
        for i in range(2):
            self.uhd_source.set_center_freq(self.freq, i)
            self.uhd_source.set_gain(40, i)

        # --- QPSK Chain (Matched to Flowchart) [cite: 61, 62, 63] ---
        self.agc_q = analog.agc_cc(1e-4, 1.0, 1.0, 4000)
        self.fll_q = digital.fll_band_edge_cc(self.sps, self.excess_bw, 44, 0.0628)
        self.sync_q = digital.pfb_clock_sync_ccf(self.sps, 0.0628, firdes.root_raised_cosine(32, 32, 1.0, 0.35, 11*32), 32, 16, 1.5, 2)
        
        # Linear Equalizer 
        self.alg_q = digital.cma_equalizer_cc(0.01, 1.0, 1) # step_size=0.01, delta=1.0 
        self.eq_q = digital.linear_equalizer(15, 2, self.alg_q) # num_taps=15, sps=2 
        
        self.costas_q = digital.costas_loop_cc(0.0628, 4, False) # QPSK [cite: 58]
        self.decoder_q = digital.constellation_decoder_cb(digital.constellation_qpsk().base())
        self.corr_q = digital.correlate_access_code_bb_ts(self.qpsk_access_code, 6, "packet_len")
        self.rec_q = ImageRecoveryBlock('qpsk_grc_fixed.jpg')

        # --- BPSK Chain (Kept Same) ---
        self.agc_b = analog.agc_cc(1e-4, 1.0, 1.0, 4000)
        self.fll_b = digital.fll_band_edge_cc(self.sps, self.excess_bw, 44, 0.0628)
        self.sync_b = digital.pfb_clock_sync_ccf(self.sps, 0.0628, firdes.root_raised_cosine(32, 32, 1.0, 0.35, 11*32), 32, 16, 1.5, 1)
        self.costas_b = digital.costas_loop_cc(0.05, 2, False)
        self.decoder_b = digital.constellation_decoder_cb(digital.constellation_bpsk().base())
        self.corr_b = digital.correlate_access_code_bb_ts(self.bpsk_access_code, 1, "packet_len")
        self.rec_b = ImageRecoveryBlock('bpsk_standard.jpg')

        # --- Connections ---
        # QPSK Flow [cite: 103]
        self.connect((self.uhd_source, 0), self.agc_q, self.fll_q, self.sync_q, self.eq_q, self.costas_q, self.decoder_q)
        self.connect(self.decoder_q, digital.diff_decoder_bb(4, digital.DIFF_DIFFERENTIAL), blocks.unpack_k_bits_bb(2), self.corr_q)
        self.connect(self.corr_q, blocks.repack_bits_bb(1, 8, "packet_len", True), digital.crc32_bb(True, "packet_len"), pdu.tagged_stream_to_pdu(gr.types.byte_t, 'packet_len'))
        self.msg_connect((self.pdu_qpsk_port := pdu.tagged_stream_to_pdu(gr.types.byte_t, 'packet_len'), 'pdus'), (self.rec_q, 'pdus'))

        # BPSK Flow
        self.connect((self.uhd_source, 1), self.agc_b, self.fll_b, self.sync_b, self.costas_b, self.decoder_b)
        self.connect(self.decoder_b, digital.diff_decoder_bb(2, digital.DIFF_DIFFERENTIAL), self.corr_b)
        self.connect(self.corr_b, blocks.repack_bits_bb(1, 8, "packet_len", True), digital.crc32_bb(True, "packet_len"), pdu.tagged_stream_to_pdu(gr.types.byte_t, 'packet_len'))
        self.msg_connect((self.pdu_bpsk_port := pdu.tagged_stream_to_pdu(gr.types.byte_t, 'packet_len'), 'pdus'), (self.rec_b, 'pdus'))

        # Re-attach PDU blocks to flowgraph
        self.connect(self.corr_q, self.pdu_qpsk_port)
        self.connect(self.corr_b, self.pdu_bpsk_port)

        self.rec_q.proxy.image_received.connect(lambda p: self.qpsk_lbl.setPixmap(Qt.QPixmap(p).scaled(400, 400, Qt.Qt.KeepAspectRatio)))
        self.rec_b.proxy.image_received.connect(lambda p: self.bpsk_lbl.setPixmap(Qt.QPixmap(p).scaled(400, 400, Qt.Qt.KeepAspectRatio)))

    def closeEvent(self, event):
        self.stop(); self.wait(); event.accept()

def main():
    qapp = Qt.QApplication(sys.argv)
    tb = dual_pole_rx(); tb.start(); tb.show()
    signal.signal(signal.SIGINT, lambda *a: Qt.QApplication.quit())
    qapp.exec_()

if __name__ == '__main__':
    main()