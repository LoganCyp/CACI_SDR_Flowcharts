#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-Pole MIMO Receiver — QPSK (ch0) + BPSK (ch1)
Fixed DSP chain, no sliders, hardcoded optimal parameters.

Signal flow per channel:
  USRP -> AGC -> FLL -> PFB(osps=1) -> MIMO XPIC -> Costas -> decode
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
        self.setWindowTitle("Dual-Pole SDR Receiver")
        self.resize(900, 500)

        layout = Qt.QHBoxLayout(self)

        ##################################################
        # Variables
        ##################################################
        sps         = 4
        nfilts      = 32
        samp_rate   = 1e6
        freq        = 2.45e9
        excess_bw   = 0.35
        rx_gain     = 50

        fll_bw      = 0.0125
        pfb_bw      = 0.02
        costas_qpsk = 0.025
        costas_bpsk = 0.02
        mimo_mu     = 1e-4
        mimo_rho    = 0.05

        qpsk_access = '10010110110110100101000111011001'
        bpsk_access = '11100001010110101110100010010011'

        ##################################################
        # GUI — two image panels only
        ##################################################

        qpsk_group = Qt.QGroupBox(f"Channel 0 — QPSK  ({freq / 1e9:.2f} GHz)")
        qpsk_vbox = Qt.QVBoxLayout()
        qpsk_group.setLayout(qpsk_vbox)
        self.qpsk_image_label = Qt.QLabel("Awaiting QPSK Image...")
        self.qpsk_image_label.setAlignment(Qt.Qt.AlignCenter)
        self.qpsk_image_label.setStyleSheet(
            "font-size: 20px; color: #555; border: 2px dashed #aaa; "
            "background: #eee;")
        self.qpsk_image_label.setMinimumSize(400, 400)
        qpsk_vbox.addWidget(self.qpsk_image_label)
        layout.addWidget(qpsk_group)

        bpsk_group = Qt.QGroupBox(f"Channel 1 — BPSK  ({freq / 1e9:.2f} GHz)")
        bpsk_vbox = Qt.QVBoxLayout()
        bpsk_group.setLayout(bpsk_vbox)
        self.bpsk_image_label = Qt.QLabel("Awaiting BPSK Image...")
        self.bpsk_image_label.setAlignment(Qt.Qt.AlignCenter)
        self.bpsk_image_label.setStyleSheet(
            "font-size: 20px; color: #555; border: 2px dashed #aaa; "
            "background: #eee;")
        self.bpsk_image_label.setMinimumSize(400, 400)
        bpsk_vbox.addWidget(self.bpsk_image_label)
        layout.addWidget(bpsk_group)

        ##################################################
        # SDR Source — 2-channel B210
        ##################################################

        self.usrp = uhd.usrp_source(
            ",".join(("", '')),
            uhd.stream_args(cpu_format="fc32", args='',
                            channels=list(range(0, 2))),
        )
        self.usrp.set_samp_rate(samp_rate)

        self.usrp.set_center_freq(freq, 0)
        self.usrp.set_antenna("RX2", 0)
        self.usrp.set_gain(rx_gain, 0)

        self.usrp.set_center_freq(freq, 1)
        self.usrp.set_antenna("RX2", 1)
        self.usrp.set_gain(rx_gain, 1)

        ##################################################
        # AGC — per channel, max_gain=65536
        ##################################################

        self.agc_0 = analog.agc_cc(1e-4, 1.0, 1.0, 65536)
        self.agc_1 = analog.agc_cc(1e-4, 1.0, 1.0, 65536)

        ##################################################
        # FLL Band Edge — frequency lock
        ##################################################

        self.fll_0 = digital.fll_band_edge_cc(sps, excess_bw, 44, fll_bw)
        self.fll_1 = digital.fll_band_edge_cc(sps, excess_bw, 44, fll_bw)

        ##################################################
        # PFB Clock Sync — timing recovery
        # taps encode sps via 1.0/sps, total = 11*sps*nfilts
        # osps = 1 on both channels
        ##################################################

        rrc_taps = firdes.root_raised_cosine(
            nfilts,                         # gain
            nfilts,                         # sampling_freq
            1.0 / float(sps),              # symbol_rate
            excess_bw,                      # rolloff
            11 * sps * nfilts               # ntaps
        )

        self.pfb_0 = digital.pfb_clock_sync_ccf(
            sps, pfb_bw, rrc_taps, nfilts, nfilts // 2, 1.5, 1)
        self.pfb_1 = digital.pfb_clock_sync_ccf(
            sps, pfb_bw, rrc_taps, nfilts, nfilts // 2, 1.5, 1)

        ##################################################
        # MIMO XPIC — post-PFB at 1 sps
        ##################################################

        self.mimo = mimo_xpic_2x2(mu=mimo_mu, rho=mimo_rho)

        ##################################################
        # Costas Loops — phase lock
        ##################################################

        self.costas_0 = digital.costas_loop_cc(costas_qpsk, 4, False)
        self.costas_1 = digital.costas_loop_cc(costas_bpsk, 2, False)

        ##################################################
        # QPSK bit recovery
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

        self.rec_q.proxy.image_received.connect(self._show_qpsk)
        self.rec_b.proxy.image_received.connect(self._show_bpsk)

        ##################################################
        # Connections
        ##################################################

        # Front end
        self.connect((self.usrp, 0), (self.agc_0, 0))
        self.connect((self.usrp, 1), (self.agc_1, 0))

        self.connect((self.agc_0, 0), (self.fll_0, 0))
        self.connect((self.agc_1, 0), (self.fll_1, 0))

        self.connect((self.fll_0, 0), (self.pfb_0, 0))
        self.connect((self.fll_1, 0), (self.pfb_1, 0))

        # MIMO at 1 sps
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

    def _show_qpsk(self, path):
        px = Qt.QPixmap(path)
        if not px.isNull():
            self.qpsk_image_label.setPixmap(
                px.scaled(self.qpsk_image_label.size(),
                          Qt.Qt.KeepAspectRatio,
                          Qt.Qt.SmoothTransformation))
            self.qpsk_image_label.setStyleSheet(
                "border: none; background: transparent;")

    def _show_bpsk(self, path):
        px = Qt.QPixmap(path)
        if not px.isNull():
            self.bpsk_image_label.setPixmap(
                px.scaled(self.bpsk_image_label.size(),
                          Qt.Qt.KeepAspectRatio,
                          Qt.Qt.SmoothTransformation))
            self.bpsk_image_label.setStyleSheet(
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