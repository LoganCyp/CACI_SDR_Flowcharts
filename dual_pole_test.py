#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-Pole MIMO Receiver — QPSK (ch0) + BPSK (ch1)
QPSK parameters match the proven working GRC flowgraph at 3 GHz.
BPSK chain uses equivalent parameters with order-2 adjustments.

Signal flow per channel:
  USRP -> AGC -> FLL -> PFB -> MIMO XPIC -> Costas -> decode
"""

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
# 1. CPU-Safe Image Recovery Block
##################################################
class ImageRecoveryBlock(gr.basic_block):
    def __init__(self, out_jpg='recovered_latest.jpg'):
        gr.basic_block.__init__(self, name="Image Recovery",
                                in_sig=None, out_sig=None)
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
        if s < 0:
            return
        e = self.buf.find(JPEG_END, s)
        if e < 0:
            return

        jpg = bytes(self.buf[s:e + 2])

        if self._valid_jpeg(jpg):
            with open(self.out_jpg, "wb") as f:
                f.write(jpg)
            self.proxy.image_received.emit(self.out_jpg)
            self.buf = self.buf[e + 2:]
        else:
            del self.buf[:s + 2]

    def _valid_jpeg(self, b):
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
# 2. NUMBA-Compiled MIMO XPIC Equalizer
##################################################
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


##################################################
# 3. Main GUI and Flowgraph
##################################################
class dual_pole_rx(gr.top_block, Qt.QWidget):
    def __init__(self):
        gr.top_block.__init__(self, "Dual-Pole Rx System",
                              catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Dual-Pole SDR Receiver — QPSK + BPSK @ 3 GHz")
        self.resize(900, 500)

        layout = Qt.QHBoxLayout(self)

        ##################################################
        # Variables — QPSK values taken directly from the
        # working GRC flowgraph (qpsk_rx_b210)
        ##################################################
        sps         = 4
        samp_rate   = 1e6
        freq        = 3e9
        excess_bw   = 0.35

        # Gains — exact from working GRC
        rx_gain_ch0 = 20          # QPSK — proven working
        rx_gain_ch1 = 20          # BPSK — same hardware, same distance

        # Access codes — different per channel to prevent
        # cross-pol false sync
        qpsk_access = '11100001010110101110100010010011'
        bpsk_access = '10010110110110100101000111011001'

        # RRC taps — EXACT formula from working GRC:
        #   firdes.root_raised_cosine(32, 32, 1.0/1.0, 0.35, 11*32)
        # This is the formula that works. Do not change it.
        rrc_taps = firdes.root_raised_cosine(32, 32, 1.0 / 1.0, 0.35, 11 * 32)

        ##################################################
        # GUI — two image panels
        ##################################################

        ch0_group = Qt.QGroupBox(f"Channel 0 — QPSK  ({freq / 1e9:.1f} GHz)")
        ch0_vbox = Qt.QVBoxLayout()
        ch0_group.setLayout(ch0_vbox)
        self.ch0_image_label = Qt.QLabel("Awaiting QPSK Image...")
        self.ch0_image_label.setAlignment(Qt.Qt.AlignCenter)
        self.ch0_image_label.setStyleSheet(
            "font-size: 20px; color: #555; border: 2px dashed #aaa; "
            "background: #eee;")
        self.ch0_image_label.setMinimumSize(400, 400)
        ch0_vbox.addWidget(self.ch0_image_label)
        layout.addWidget(ch0_group)

        ch1_group = Qt.QGroupBox(f"Channel 1 — BPSK  ({freq / 1e9:.1f} GHz)")
        ch1_vbox = Qt.QVBoxLayout()
        ch1_group.setLayout(ch1_vbox)
        self.ch1_image_label = Qt.QLabel("Awaiting BPSK Image...")
        self.ch1_image_label.setAlignment(Qt.Qt.AlignCenter)
        self.ch1_image_label.setStyleSheet(
            "font-size: 20px; color: #555; border: 2px dashed #aaa; "
            "background: #eee;")
        self.ch1_image_label.setMinimumSize(400, 400)
        ch1_vbox.addWidget(self.ch1_image_label)
        layout.addWidget(ch1_group)

        ##################################################
        # USRP Source — 2-channel B210 at 3 GHz
        ##################################################

        self.usrp = uhd.usrp_source(
            ",".join(("", '')),
            uhd.stream_args(cpu_format="fc32", args='',
                            channels=list(range(0, 2))),
        )
        self.usrp.set_samp_rate(samp_rate)

        self.usrp.set_center_freq(freq, 0)
        self.usrp.set_antenna("RX2", 0)
        self.usrp.set_gain(rx_gain_ch0, 0)

        self.usrp.set_center_freq(freq, 1)
        self.usrp.set_antenna("RX2", 1)
        self.usrp.set_gain(rx_gain_ch1, 1)

        ##################################################
        # AGC — exact from working GRC
        # rate=1e-4, reference=1.0, gain=1.0, max_gain=4000
        ##################################################

        self.agc_0 = analog.agc_cc(1e-4, 1.0, 1.0, 4000)
        self.agc_1 = analog.agc_cc(1e-4, 1.0, 1.0, 4000)

        ##################################################
        # FLL Band Edge — exact from working GRC
        # sps=4, excess_bw=0.35, filter_size=44, w=0.0628
        ##################################################

        self.fll_0 = digital.fll_band_edge_cc(sps, excess_bw, 44, 0.0628)
        self.fll_1 = digital.fll_band_edge_cc(sps, excess_bw, 44, 0.0628)

        ##################################################
        # PFB Clock Sync — exact from working GRC
        # sps=4, loop_bw=0.0628, taps=rrc_taps,
        # filter_size=32, init_phase=16, max_dev=1.5,
        # osps=2
        #
        # Your GRC uses osps=2 and it works. Keeping it
        # identical. The MIMO block at 2 sps is fine
        # because it's a sample-by-sample operation.
        ##################################################

        self.pfb_0 = digital.pfb_clock_sync_ccf(
            sps, 0.0628, rrc_taps, 32, 16, 1.5, 2)
        self.pfb_1 = digital.pfb_clock_sync_ccf(
            sps, 0.0628, rrc_taps, 32, 16, 1.5, 2)

        ##################################################
        # MIMO XPIC — post-PFB
        # At 2 sps from PFB, MIMO processes every sample.
        # CMA still converges because the on-symbol samples
        # dominate the error signal. The cross-pol term
        # works at any sample rate.
        ##################################################

        self.mimo = mimo_xpic_2x2(mu=1e-4, rho=0.05)

        ##################################################
        # Costas Loops — exact from working GRC
        # QPSK: w=0.0628, order=4
        # BPSK: w=0.0628, order=2
        ##################################################

        self.costas_0 = digital.costas_loop_cc(0.0628, 4, False)
        self.costas_1 = digital.costas_loop_cc(0.0628, 2, False)

        ##################################################
        # QPSK bit recovery — exact from working GRC
        ##################################################

        self.dec_q   = digital.constellation_decoder_cb(
                           digital.constellation_qpsk().base())
        self.diff_q  = digital.diff_decoder_bb(4, digital.DIFF_DIFFERENTIAL)
        self.unp_q   = blocks.unpack_k_bits_bb(2)
        self.corr_q  = digital.correlate_access_code_bb_ts(
                           qpsk_access, 2, "packet_len")
        self.rep_q   = blocks.repack_bits_bb(
                           1, 8, "packet_len", True, gr.GR_MSB_FIRST)
        self.crc_q   = digital.crc32_bb(True, "packet_len", True)
        self.pdu_q   = pdu.tagged_stream_to_pdu(
                           gr.types.byte_t, 'packet_len')
        self.rec_q   = ImageRecoveryBlock(out_jpg='qpsk_recovered.jpg')

        ##################################################
        # BPSK bit recovery
        # Same structure but order-2, no unpack_k_bits
        ##################################################

        self.dec_b   = digital.constellation_decoder_cb(
                           digital.constellation_bpsk().base())
        self.diff_b  = digital.diff_decoder_bb(2, digital.DIFF_DIFFERENTIAL)
        self.corr_b  = digital.correlate_access_code_bb_ts(
                           bpsk_access, 2, "packet_len")
        self.rep_b   = blocks.repack_bits_bb(
                           1, 8, "packet_len", True, gr.GR_MSB_FIRST)
        self.crc_b   = digital.crc32_bb(True, "packet_len", True)
        self.pdu_b   = pdu.tagged_stream_to_pdu(
                           gr.types.byte_t, 'packet_len')
        self.rec_b   = ImageRecoveryBlock(out_jpg='bpsk_recovered.jpg')

        ##################################################
        # Qt image signals
        ##################################################

        self.rec_q.proxy.image_received.connect(self._show_ch0)
        self.rec_b.proxy.image_received.connect(self._show_ch1)

        ##################################################
        # Connections
        # USRP -> AGC -> FLL -> PFB(2sps) -> MIMO -> Costas -> decode
        ##################################################

        # Front end
        self.connect((self.usrp, 0), (self.agc_0, 0))
        self.connect((self.usrp, 1), (self.agc_1, 0))

        self.connect((self.agc_0, 0), (self.fll_0, 0))
        self.connect((self.agc_1, 0), (self.fll_1, 0))

        self.connect((self.fll_0, 0), (self.pfb_0, 0))
        self.connect((self.fll_1, 0), (self.pfb_1, 0))

        # MIMO at 2 sps (matching working GRC osps)
        self.connect((self.pfb_0, 0), (self.mimo, 0))
        self.connect((self.pfb_1, 0), (self.mimo, 1))

        self.connect((self.mimo, 0), (self.costas_0, 0))
        self.connect((self.mimo, 1), (self.costas_1, 0))

        # QPSK decode
        self.connect((self.costas_0, 0), (self.dec_q, 0))
        self.connect((self.dec_q, 0),    (self.diff_q, 0))
        self.connect((self.diff_q, 0),   (self.unp_q, 0))
        self.connect((self.unp_q, 0),    (self.corr_q, 0))
        self.connect((self.corr_q, 0),   (self.rep_q, 0))
        self.connect((self.rep_q, 0),    (self.crc_q, 0))
        self.connect((self.crc_q, 0),    (self.pdu_q, 0))
        self.msg_connect((self.pdu_q, 'pdus'), (self.rec_q, 'pdus'))

        # BPSK decode
        self.connect((self.costas_1, 0), (self.dec_b, 0))
        self.connect((self.dec_b, 0),    (self.diff_b, 0))
        self.connect((self.diff_b, 0),   (self.corr_b, 0))
        self.connect((self.corr_b, 0),   (self.rep_b, 0))
        self.connect((self.rep_b, 0),    (self.crc_b, 0))
        self.connect((self.crc_b, 0),    (self.pdu_b, 0))
        self.msg_connect((self.pdu_b, 'pdus'), (self.rec_b, 'pdus'))

    ##################################################
    # Image display
    ##################################################

    def _show_ch0(self, path):
        px = Qt.QPixmap(path)
        if not px.isNull():
            self.ch0_image_label.setPixmap(
                px.scaled(self.ch0_image_label.size(),
                          Qt.Qt.KeepAspectRatio,
                          Qt.Qt.SmoothTransformation))
            self.ch0_image_label.setStyleSheet(
                "border: none; background: transparent;")

    def _show_ch1(self, path):
        px = Qt.QPixmap(path)
        if not px.isNull():
            self.ch1_image_label.setPixmap(
                px.scaled(self.ch1_image_label.size(),
                          Qt.Qt.KeepAspectRatio,
                          Qt.Qt.SmoothTransformation))
            self.ch1_image_label.setStyleSheet(
                "border: none; background: transparent;")

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