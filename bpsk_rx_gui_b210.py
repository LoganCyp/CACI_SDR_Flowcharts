#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dual-Pole BPSK Receiver for B210
=================================
Matched to dual_pole_tx_b210_bpsk.py:
  - Ch0 BPSK, access code: 10010110110110100101000111011001
  - Ch1 BPSK, access code: 11100001010110101110100010010011
  - Both: differential encoding, 4 sps, excess_bw=0.35, 1 Msps, 2.4 GHz
  - Packet: 60-byte payload + CRC32 + header_format_default framing
"""

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
    packet_received = pyqtSignal(int)


##################################################
# Image Recovery Block (with packet counter)
##################################################
class ImageRecoveryBlock(gr.basic_block):
    def __init__(self, out_jpg='recovered_latest.jpg'):
        gr.basic_block.__init__(self, name="Image Recovery", in_sig=None, out_sig=None)
        self.message_port_register_in(pmt.intern("pdus"))
        self.set_msg_handler(pmt.intern("pdus"), self._handle)

        self.out_jpg = out_jpg
        self.buf = bytearray()
        self.proxy = SignalProxy()
        self.pkt_count = 0

    def _handle(self, msg):
        vec = pmt.cdr(msg)
        data = bytes(pmt.u8vector_elements(vec))
        self.buf.extend(data)

        self.pkt_count += 1
        self.proxy.packet_received.emit(self.pkt_count)

        if len(self.buf) > MAX_BUF_SIZE:
            self.buf = self.buf[-MAX_BUF_SIZE:]

        # Extract all complete JPEGs in the buffer
        while True:
            s = self.buf.find(JPEG_START)
            if s < 0:
                break
            e = self.buf.find(JPEG_END, s + 2)
            if e < 0:
                # Trim anything before the start marker to keep buffer lean
                if s > 0:
                    del self.buf[:s]
                break

            jpg = bytes(self.buf[s:e + 2])
            self.buf = bytearray(self.buf[e + 2:])

            if self._valid_jpeg(jpg):
                with open(self.out_jpg, "wb") as f:
                    f.write(jpg)
                self.proxy.image_received.emit(self.out_jpg)

    @staticmethod
    def _valid_jpeg(b):
        if len(b) < 100:
            return False
        if not HAS_PIL:
            return True
        try:
            im = Image.open(io.BytesIO(b))
            im.verify()
            return True
        except Exception:
            return False


##################################################
# Dual-Pole BPSK Receiver
##################################################
class dual_pole_rx_bpsk(gr.top_block, Qt.QWidget):
    def __init__(self):
        gr.top_block.__init__(self, "Dual-Pole BPSK Receiver", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Dual-Pole BPSK Receiver (B210)")
        self.resize(1200, 650)
        self.main_layout = Qt.QHBoxLayout(self)

        ##################################################
        # Parameters — matched to TX
        ##################################################
        self.sps = 4
        self.samp_rate = 1e6
        self.freq = 2.4e9
        self.excess_bw = 0.35

        self.ch0_access_code = '10010110110110100101000111011001'
        self.ch1_access_code = '11100001010110101110100010010011'
        self.gain_ch0 = 40
        self.gain_ch1 = 40

        # Costas loop order for BPSK on both channels
        self.costas_order = 2

        ##################################################
        # GUI
        ##################################################
        self._build_gui()

        ##################################################
        # SDR Source — 2-channel MIMO
        ##################################################
        self.uhd_usrp_source = uhd.usrp_source(
            ",".join(("", "num_recv_frames=128")),
            uhd.stream_args(cpu_format="fc32", args='', channels=[0, 1]),
        )
        self.uhd_usrp_source.set_samp_rate(self.samp_rate)
        for ch in range(2):
            self.uhd_usrp_source.set_center_freq(self.freq, ch)
            self.uhd_usrp_source.set_antenna("TX/RX", ch)
        self.uhd_usrp_source.set_gain(self.gain_ch0, 0)
        self.uhd_usrp_source.set_gain(self.gain_ch1, 1)

        ##################################################
        # Shared RRC taps (computed once, used by both)
        ##################################################
        nfilts = 32
        ntaps = 11 * nfilts * self.sps
        rrc_taps = firdes.root_raised_cosine(nfilts, nfilts, 1.0, self.excess_bw, ntaps)

        ##################################################
        # Channel 0 — BPSK Rx chain
        ##################################################
        self.agc_0 = analog.agc_cc(1e-4, 1.0, 1.0, 4000)
        self.fll_0 = digital.fll_band_edge_cc(self.sps, self.excess_bw, 44, 0.0628)
        self.sync_0 = digital.pfb_clock_sync_ccf(self.sps, 0.0628, rrc_taps, nfilts, nfilts // 2, 1.5, 1)
        self.costas_0 = digital.costas_loop_cc(0.0628, self.costas_order, False)

        self.decoder_0 = digital.constellation_decoder_cb(digital.constellation_bpsk().base())
        self.diff_0 = digital.diff_decoder_bb(2, digital.DIFF_DIFFERENTIAL)
        self.corr_0 = digital.correlate_access_code_bb_ts(self.ch0_access_code, 1, "packet_len")
        self.repack_0 = blocks.repack_bits_bb(1, 8, "packet_len", True, gr.GR_MSB_FIRST)
        self.crc_0 = digital.crc32_bb(True, "packet_len", True)
        self.pdu_0 = pdu.tagged_stream_to_pdu(gr.types.byte_t, "packet_len")
        self.rec_0 = ImageRecoveryBlock(out_jpg="bpsk_ch0_recovered.jpg")

        ##################################################
        # Channel 1 — BPSK Rx chain
        ##################################################
        self.agc_1 = analog.agc_cc(1e-4, 1.0, 1.0, 4000)
        self.fll_1 = digital.fll_band_edge_cc(self.sps, self.excess_bw, 44, 0.0628)
        self.sync_1 = digital.pfb_clock_sync_ccf(self.sps, 0.0628, rrc_taps, nfilts, nfilts // 2, 1.5, 1)
        self.costas_1 = digital.costas_loop_cc(0.0628, self.costas_order, False)

        self.decoder_1 = digital.constellation_decoder_cb(digital.constellation_bpsk().base())
        self.diff_1 = digital.diff_decoder_bb(2, digital.DIFF_DIFFERENTIAL)
        self.corr_1 = digital.correlate_access_code_bb_ts(self.ch1_access_code, 1, "packet_len")
        self.repack_1 = blocks.repack_bits_bb(1, 8, "packet_len", True, gr.GR_MSB_FIRST)
        self.crc_1 = digital.crc32_bb(True, "packet_len", True)
        self.pdu_1 = pdu.tagged_stream_to_pdu(gr.types.byte_t, "packet_len")
        self.rec_1 = ImageRecoveryBlock(out_jpg="bpsk_ch1_recovered.jpg")

        ##################################################
        # Qt Signal connections
        ##################################################
        self.rec_0.proxy.image_received.connect(self.update_ch0_image)
        self.rec_0.proxy.packet_received.connect(lambda n: self.ch0_pkt_label.setText(f"Packets: {n}"))
        self.rec_1.proxy.image_received.connect(self.update_ch1_image)
        self.rec_1.proxy.packet_received.connect(lambda n: self.ch1_pkt_label.setText(f"Packets: {n}"))

        ##################################################
        # Connections — Channel 0
        ##################################################
        self.connect((self.uhd_usrp_source, 0), self.agc_0, self.fll_0, self.sync_0, self.costas_0)
        self.connect(self.costas_0, self.decoder_0, self.diff_0, self.corr_0,
                     self.repack_0, self.crc_0, self.pdu_0)
        self.msg_connect((self.pdu_0, "pdus"), (self.rec_0, "pdus"))

        ##################################################
        # Connections — Channel 1
        ##################################################
        self.connect((self.uhd_usrp_source, 1), self.agc_1, self.fll_1, self.sync_1, self.costas_1)
        self.connect(self.costas_1, self.decoder_1, self.diff_1, self.corr_1,
                     self.repack_1, self.crc_1, self.pdu_1)
        self.msg_connect((self.pdu_1, "pdus"), (self.rec_1, "pdus"))

    ##################################################
    # GUI Builder
    ##################################################
    def _build_gui(self):
        for ch, label_prefix, access_code in [
            (0, "Ch0 BPSK", self.ch0_access_code),
            (1, "Ch1 BPSK", self.ch1_access_code),
        ]:
            group = Qt.QGroupBox(f"Channel {ch}: BPSK Receiver ({self.freq / 1e9:.1f} GHz)")
            layout = Qt.QVBoxLayout()
            group.setLayout(layout)

            img_label = Qt.QLabel(f"Awaiting {label_prefix}...")
            img_label.setMinimumSize(400, 400)
            img_label.setAlignment(Qt.Qt.AlignCenter)
            img_label.setStyleSheet("border: 1px solid #444; background: #1a1a1a;")
            layout.addWidget(img_label)

            pkt_label = Qt.QLabel("Packets: 0")
            pkt_label.setStyleSheet("font-weight: bold;")
            layout.addWidget(pkt_label)

            gain_val = self.gain_ch0 if ch == 0 else self.gain_ch1
            gain_label = Qt.QLabel(f"<b>Gain:</b> {gain_val} dB")
            layout.addWidget(gain_label)

            slider = Qt.QSlider(Qt.Qt.Horizontal)
            slider.setRange(0, 76)
            slider.setValue(gain_val)
            layout.addWidget(slider)

            self.main_layout.addWidget(group)

            if ch == 0:
                self.ch0_image_label = img_label
                self.ch0_pkt_label = pkt_label
                self.ch0_gain_label = gain_label
                self.ch0_gain_slider = slider
                slider.valueChanged.connect(self.set_ch0_gain)
            else:
                self.ch1_image_label = img_label
                self.ch1_pkt_label = pkt_label
                self.ch1_gain_label = gain_label
                self.ch1_gain_slider = slider
                slider.valueChanged.connect(self.set_ch1_gain)

    ##################################################
    # Callbacks
    ##################################################
    def set_ch0_gain(self, v):
        self.gain_ch0 = v
        self.ch0_gain_label.setText(f"<b>Gain:</b> {v} dB")
        self.uhd_usrp_source.set_gain(v, 0)

    def set_ch1_gain(self, v):
        self.gain_ch1 = v
        self.ch1_gain_label.setText(f"<b>Gain:</b> {v} dB")
        self.uhd_usrp_source.set_gain(v, 1)

    def update_ch0_image(self, path):
        pm = Qt.QPixmap(path)
        if not pm.isNull():
            self.ch0_image_label.setPixmap(pm.scaled(400, 400, Qt.Qt.KeepAspectRatio, Qt.Qt.SmoothTransformation))

    def update_ch1_image(self, path):
        pm = Qt.QPixmap(path)
        if not pm.isNull():
            self.ch1_image_label.setPixmap(pm.scaled(400, 400, Qt.Qt.KeepAspectRatio, Qt.Qt.SmoothTransformation))

    def closeEvent(self, event):
        self.stop()
        self.wait()
        event.accept()


def main():
    qapp = Qt.QApplication(sys.argv)
    tb = dual_pole_rx_bpsk()
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