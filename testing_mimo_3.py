#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import signal
import io
import numpy as np
import pmt
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
# 1. CPU-Safe Image Recovery Block
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
        self.buf.extend(bytes(pmt.u8vector_elements(vec)))

        if len(self.buf) > MAX_BUF_SIZE:
            self.buf = self.buf[-MAX_BUF_SIZE:]

        s = self.buf.find(JPEG_START)
        if s < 0: return
        e = self.buf.find(JPEG_END, s)
        if e < 0: return

        jpg = bytes(self.buf[s:e+2])

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
# 2. Main GUI and Flowgraph (MIMO BYPASSED)
##################################################
class dual_pole_rx(gr.top_block, Qt.QWidget):
    def __init__(self):
        gr.top_block.__init__(self, "Dual-Pole Rx System (No MIMO)", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Dual-Pole SDR Receiver (SISO Mode Testing)")
        self.resize(1200, 600)
        self.main_layout = Qt.QHBoxLayout(self)

        self.sps = 4
        self.samp_rate = 1e6 
        self.freq = 2.4e9  
        self.excess_bw = 0.35 
        
        self.qpsk_access_code = '10010110110110100101000111011001'
        self.bpsk_access_code = '11100001010110101110100010010011'
        self.gain_ch0 = 30
        self.gain_ch1 = 30

        # --- GUI Setup ---
        self.qpsk_group = Qt.QGroupBox(f"Channel 0: QPSK Receiver ({self.freq/1e9} GHz)")
        self.qpsk_layout = Qt.QVBoxLayout()
        self.qpsk_group.setLayout(self.qpsk_layout)
        self.qpsk_image_label = Qt.QLabel("Awaiting QPSK...")
        self.qpsk_image_label.setMinimumSize(400, 400)
        self.qpsk_layout.addWidget(self.qpsk_image_label)
        self.qpsk_gain_slider = Qt.QSlider(Qt.Qt.Horizontal)
        self.qpsk_gain_slider.setRange(0, 90); self.qpsk_gain_slider.setValue(self.gain_ch0)
        self.qpsk_gain_slider.valueChanged.connect(self.set_ch0_gain)
        self.qpsk_layout.addWidget(self.qpsk_gain_slider)
        self.main_layout.addWidget(self.qpsk_group)

        self.bpsk_group = Qt.QGroupBox(f"Channel 1: BPSK Receiver ({self.freq/1e9} GHz)")
        self.bpsk_layout = Qt.QVBoxLayout()
        self.bpsk_group.setLayout(self.bpsk_layout)
        self.bpsk_image_label = Qt.QLabel("Awaiting BPSK...")
        self.bpsk_image_label.setMinimumSize(400, 400)
        self.bpsk_layout.addWidget(self.bpsk_image_label)
        self.bpsk_gain_slider = Qt.QSlider(Qt.Qt.Horizontal)
        self.bpsk_gain_slider.setRange(0, 90); self.bpsk_gain_slider.setValue(self.gain_ch1)
        self.bpsk_gain_slider.valueChanged.connect(self.set_ch1_gain)
        self.bpsk_layout.addWidget(self.bpsk_gain_slider)
        self.main_layout.addWidget(self.bpsk_group)

        # --- SDR Blocks ---
        self.uhd_usrp_source_0 = uhd.usrp_source(",".join(("", 'num_recv_frames=128')),
            uhd.stream_args(cpu_format="fc32", args='', channels=list(range(0,2))))
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)
        for i in range(2):
            self.uhd_usrp_source_0.set_center_freq(self.freq, i)
            self.uhd_usrp_source_0.set_antenna("TX/RX", i)
            self.uhd_usrp_source_0.set_gain(self.gain_ch0 if i==0 else self.gain_ch1, i)

        self.agc_0 = analog.agc_cc(1e-4, 1.0, 1.0, 4000)
        self.agc_1 = analog.agc_cc(1e-4, 1.0, 1.0, 4000)
        self.fll_0 = digital.fll_band_edge_cc(self.sps, self.excess_bw, 45, 0.05)
        self.fll_1 = digital.fll_band_edge_cc(self.sps, self.excess_bw, 44, 0.0628)
        
        rrc_taps = firdes.root_raised_cosine(32, 32, 1.0, self.excess_bw, 11*32)
        self.sync_0 = digital.pfb_clock_sync_ccf(self.sps, 0.0628, rrc_taps, 32, 16, 1.5, 2)
        self.sync_1 = digital.pfb_clock_sync_ccf(self.sps, 0.0628, rrc_taps, 32, 16, 1.5, 1)

        # --- ADDED: QPSK Linear Equalizer & CMA Algorithm ---
        self.qpsk_const = digital.constellation_qpsk()
        self.alg_qpsk = digital.adaptive_algorithm_cma(self.qpsk_const.base(), 0.01, 1.0)
        self.eq_qpsk = digital.linear_equalizer(15, 2, self.alg_qpsk, True, [], 'corr_est')
        # ----------------------------------------------------

        self.costas_0 = digital.costas_loop_cc(0.05, 4, False)
        self.costas_1 = digital.costas_loop_cc(0.05, 2, False)

        # --- QPSK Chain ---
        self.decoder_qpsk = digital.constellation_decoder_cb(self.qpsk_const.base())
        self.diff_qpsk = digital.diff_decoder_bb(4, digital.DIFF_DIFFERENTIAL)
        self.unpack_qpsk = blocks.unpack_k_bits_bb(2)
        self.corr_qpsk = digital.correlate_access_code_bb_ts(self.qpsk_access_code, 6, "packet_len")
        self.repack_qpsk = blocks.repack_bits_bb(1, 8, "packet_len", True, gr.GR_MSB_FIRST)
        self.crc_qpsk = digital.crc32_bb(True, "packet_len", True)
        self.pdu_qpsk = pdu.tagged_stream_to_pdu(gr.types.byte_t, 'packet_len')
        self.rec_qpsk = ImageRecoveryBlock(out_jpg='qpsk_no_mimo.jpg')

        # --- BPSK Chain ---
        self.bpsk_const = digital.constellation_bpsk()
        self.decoder_bpsk = digital.constellation_decoder_cb(self.bpsk_const.base())
        self.diff_bpsk = digital.diff_decoder_bb(2, digital.DIFF_DIFFERENTIAL)
        self.corr_bpsk = digital.correlate_access_code_bb_ts(self.bpsk_access_code, 1, "packet_len")
        self.repack_bpsk = blocks.repack_bits_bb(1, 8, "packet_len", True, gr.GR_MSB_FIRST)
        self.crc_bpsk = digital.crc32_bb(True, "packet_len", True)
        self.pdu_bpsk = pdu.tagged_stream_to_pdu(gr.types.byte_t, 'packet_len')
        self.rec_bpsk = ImageRecoveryBlock(out_jpg='bpsk_no_mimo.jpg')

        self.rec_qpsk.proxy.image_received.connect(self.update_qpsk_image)
        self.rec_bpsk.proxy.image_received.connect(self.update_bpsk_image)

        # --- Connections ---
        # QPSK Flow (Channel 0) - Flowing through eq_qpsk
        self.connect((self.uhd_usrp_source_0, 0), self.agc_0, self.fll_0, self.sync_0, self.eq_qpsk, self.costas_0)
        self.connect(self.costas_0, self.decoder_qpsk, self.diff_qpsk, self.unpack_qpsk, self.corr_qpsk, self.repack_qpsk, self.crc_qpsk, self.pdu_qpsk)
        self.msg_connect((self.pdu_qpsk, 'pdus'), (self.rec_qpsk, 'pdus'))

        # BPSK Flow (Channel 1) - Standard SISO Flow
        self.connect((self.uhd_usrp_source_0, 1), self.agc_1, self.fll_1, self.sync_1, self.costas_1)
        self.connect(self.costas_1, self.decoder_bpsk, self.diff_bpsk, self.corr_bpsk, self.repack_bpsk, self.crc_bpsk, self.pdu_bpsk)
        self.msg_connect((self.pdu_bpsk, 'pdus'), (self.rec_bpsk, 'pdus'))

    def set_ch0_gain(self, v): self.uhd_usrp_source_0.set_gain(v, 0)
    def set_ch1_gain(self, v): self.uhd_usrp_source_0.set_gain(v, 1)

    def update_qpsk_image(self, path):
        pm = Qt.QPixmap(path)
        if not pm.isNull(): self.qpsk_image_label.setPixmap(pm.scaled(400, 400, Qt.Qt.KeepAspectRatio))

    def update_bpsk_image(self, path):
        pm = Qt.QPixmap(path)
        if not pm.isNull(): self.bpsk_image_label.setPixmap(pm.scaled(400, 400, Qt.Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.stop(); self.wait(); event.accept()

def main():
    qapp = Qt.QApplication(sys.argv)
    tb = dual_pole_rx(); tb.start(); tb.show()
    signal.signal(signal.SIGINT, lambda *a: Qt.QApplication.quit())
    qapp.exec_()

if __name__ == '__main__':
    main()