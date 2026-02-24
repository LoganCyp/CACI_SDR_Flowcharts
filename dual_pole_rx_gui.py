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

class SignalProxy(QObject):
    image_received = pyqtSignal(str)

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
# CPU-Optimized MIMO Equalizer
##################################################
class mimo_cma_2x2(gr.sync_block):
    def __init__(self, mu=1e-4):
        gr.sync_block.__init__(self,
            name="MIMO CMA XPIC",
            in_sig=[np.complex64, np.complex64],
            out_sig=[np.complex64, np.complex64])
        
        self.mu = mu
        # Store weights as raw scalars instead of an array to prevent overhead
        self.w00 = 1.0 + 0j
        self.w01 = 0.0 + 0j
        self.w10 = 0.0 + 0j
        self.w11 = 1.0 + 0j

    def work(self, input_items, output_items):
        in0, in1 = input_items[0], input_items[1]
        out0, out1 = output_items[0], output_items[1]
        
        # Load state into ultra-fast local variables
        mu = self.mu
        w00, w01, w10, w11 = self.w00, self.w01, self.w10, self.w11
        
        n_samples = len(in0)
        
        for i in range(n_samples):
            x0 = in0[i]
            x1 = in1[i]
            
            # 1. Unrolled Dot Product (Mixing)
            y0 = w00 * x0 + w01 * x1
            y1 = w10 * x0 + w11 * x1
            
            out0[i] = y0
            out1[i] = y1
            
            # 2. Fast Magnitude Squared calculation (avoids np.abs overhead)
            mag_sq0 = y0.real**2 + y0.imag**2
            mag_sq1 = y1.real**2 + y1.imag**2
            
            # 3. Fast Safety Catch (mag_sq0 != mag_sq0 checks for NaN natively)
            if mag_sq0 > 10.0 or mag_sq1 > 10.0 or mag_sq0 != mag_sq0:
                w00, w01 = 1.0+0j, 0.0+0j
                w10, w11 = 0.0+0j, 1.0+0j
                continue
            
            # 4. Error Calculation
            err0 = y0 * (mag_sq0 - 1.0)
            err1 = y1 * (mag_sq1 - 1.0)
            
            # 5. Conjugate x
            x0_c = x0.conjugate()
            x1_c = x1.conjugate()
            
            # 6. Unrolled Update (avoids np.outer overhead)
            w00 -= mu * err0 * x0_c
            w01 -= mu * err0 * x1_c
            w10 -= mu * err1 * x0_c
            w11 -= mu * err1 * x1_c

        # Save state for the next chunk of samples
        self.w00, self.w01, self.w10, self.w11 = w00, w01, w10, w11
        
        return n_samples

##################################################
# Main GUI
##################################################
class dual_pole_rx(gr.top_block, Qt.QWidget):
    def __init__(self):
        gr.top_block.__init__(self, "Dual-Pole Rx System", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Dual-Pole SDR Receiver (2.4 GHz MIMO)")
        self.resize(1200, 600)

        self.main_layout = Qt.QHBoxLayout(self)

        self.sps = 4
        self.samp_rate = 100e3  # LOWERED TO 100 ksps
        self.freq = 2.4e9  
        self.excess_bw = 0.35  
        
        self.qpsk_access_code = '10010110110110100101000111011001'
        self.bpsk_access_code = '11100001010110101110100010010011'
        
        self.gain_ch0 = 30
        self.gain_ch1 = 30

        # --- Channel 0: QPSK UI ---
        self.qpsk_group = Qt.QGroupBox(f"Channel 0: QPSK Receiver ({self.freq/1e9} GHz)")
        self.qpsk_layout = Qt.QVBoxLayout()
        self.qpsk_group.setLayout(self.qpsk_layout)
        
        self.qpsk_image_label = Qt.QLabel("Awaiting QPSK Image...")
        self.qpsk_image_label.setAlignment(Qt.Qt.AlignCenter)
        self.qpsk_image_label.setStyleSheet("font-size: 20px; color: #555; border: 2px dashed #aaa; background: #eee;")
        self.qpsk_image_label.setMinimumSize(400, 400)
        self.qpsk_layout.addWidget(self.qpsk_image_label)
        
        self.qpsk_gain_label = Qt.QLabel(f"<b>RX 0 Gain:</b> {self.gain_ch0} dB")
        self.qpsk_layout.addWidget(self.qpsk_gain_label)
        self.qpsk_gain_slider = Qt.QSlider(Qt.Qt.Horizontal)
        self.qpsk_gain_slider.setRange(0, 90)
        self.qpsk_gain_slider.setValue(self.gain_ch0)
        self.qpsk_gain_slider.valueChanged.connect(self.set_ch0_gain)
        self.qpsk_layout.addWidget(self.qpsk_gain_slider)
        
        self.main_layout.addWidget(self.qpsk_group)

        # --- Channel 1: BPSK UI ---
        self.bpsk_group = Qt.QGroupBox(f"Channel 1: BPSK Receiver ({self.freq/1e9} GHz)")
        self.bpsk_layout = Qt.QVBoxLayout()
        self.bpsk_group.setLayout(self.bpsk_layout)
        
        self.bpsk_image_label = Qt.QLabel("Awaiting BPSK Image...")
        self.bpsk_image_label.setAlignment(Qt.Qt.AlignCenter)
        self.bpsk_image_label.setStyleSheet("font-size: 20px; color: #555; border: 2px dashed #aaa; background: #eee;")
        self.bpsk_image_label.setMinimumSize(400, 400)
        self.bpsk_layout.addWidget(self.bpsk_image_label)
        
        self.bpsk_gain_label = Qt.QLabel(f"<b>RX 1 Gain:</b> {self.gain_ch1} dB")
        self.bpsk_layout.addWidget(self.bpsk_gain_label)
        self.bpsk_gain_slider = Qt.QSlider(Qt.Qt.Horizontal)
        self.bpsk_gain_slider.setRange(0, 90)
        self.bpsk_gain_slider.setValue(self.gain_ch1)
        self.bpsk_gain_slider.valueChanged.connect(self.set_ch1_gain)
        self.bpsk_layout.addWidget(self.bpsk_gain_slider)
        
        self.main_layout.addWidget(self.bpsk_group)

        # --- SDR Blocks ---
        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join(("", '')),
            uhd.stream_args(cpu_format="fc32", args='', channels=list(range(0,2))),
        )
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)
        
        self.uhd_usrp_source_0.set_center_freq(self.freq, 0)
        self.uhd_usrp_source_0.set_antenna("TX/RX", 0)  
        self.uhd_usrp_source_0.set_gain(self.gain_ch0, 0)
        
        self.uhd_usrp_source_0.set_center_freq(self.freq, 1)
        self.uhd_usrp_source_0.set_antenna("TX/RX", 1)  
        self.uhd_usrp_source_0.set_gain(self.gain_ch1, 1)

        self.agc_0 = analog.agc_cc(1e-4, 1.0, 1.0, 4000)
        self.agc_1 = analog.agc_cc(1e-4, 1.0, 1.0, 4000)
        
        self.fll_0 = digital.fll_band_edge_cc(self.sps, self.excess_bw, 45, 0.05)
        self.fll_1 = digital.fll_band_edge_cc(self.sps, self.excess_bw, 44, 0.0628)
        
        rrc_taps = firdes.root_raised_cosine(32, 32, 1.0, self.excess_bw, 11*32)
        self.sync_0 = digital.pfb_clock_sync_ccf(self.sps, 0.0628, rrc_taps, 32, 16, 1.5, 2)
        self.sync_1 = digital.pfb_clock_sync_ccf(self.sps, 0.0628, rrc_taps, 32, 16, 1.5, 1)

        self.mimo_eq = mimo_cma_2x2()

        self.decoder_qpsk = digital.constellation_decoder_cb(digital.constellation_qpsk().base())
        self.diff_qpsk = digital.diff_decoder_bb(4, digital.DIFF_DIFFERENTIAL)
        self.unpack_qpsk = blocks.unpack_k_bits_bb(2)
        self.corr_qpsk = digital.correlate_access_code_bb_ts(self.qpsk_access_code, 6, "packet_len")
        self.repack_qpsk = blocks.repack_bits_bb(1, 8, "packet_len", True, gr.GR_MSB_FIRST)
        self.crc_qpsk = digital.crc32_bb(True, "packet_len", True)
        self.pdu_qpsk = pdu.tagged_stream_to_pdu(gr.types.byte_t, 'packet_len')
        self.rec_qpsk = ImageRecoveryBlock(out_jpg='qpsk_recovered.jpg')

        self.decoder_bpsk = digital.constellation_decoder_cb(digital.constellation_bpsk().base())
        self.diff_bpsk = digital.diff_decoder_bb(2, digital.DIFF_DIFFERENTIAL)
        self.corr_bpsk = digital.correlate_access_code_bb_ts(self.bpsk_access_code, 1, "packet_len")
        self.repack_bpsk = blocks.repack_bits_bb(1, 8, "packet_len", True, gr.GR_MSB_FIRST)
        self.crc_bpsk = digital.crc32_bb(True, "packet_len", True)
        self.pdu_bpsk = pdu.tagged_stream_to_pdu(gr.types.byte_t, 'packet_len')
        self.rec_bpsk = ImageRecoveryBlock(out_jpg='bpsk_recovered.jpg')

        self.rec_qpsk.proxy.image_received.connect(self.update_qpsk_image)
        self.rec_bpsk.proxy.image_received.connect(self.update_bpsk_image)

        # --- Connections ---
        self.connect((self.uhd_usrp_source_0, 0), (self.agc_0, 0))
        self.connect((self.uhd_usrp_source_0, 1), (self.agc_1, 0))
        
        self.connect((self.agc_0, 0), (self.fll_0, 0))
        self.connect((self.agc_1, 0), (self.fll_1, 0))
        
        self.connect((self.fll_0, 0), (self.sync_0, 0))
        self.connect((self.fll_1, 0), (self.sync_1, 0))
        
        self.connect((self.sync_0, 0), (self.mimo_eq, 0))
        self.connect((self.sync_1, 0), (self.mimo_eq, 1))

        self.connect((self.mimo_eq, 0), (self.decoder_qpsk, 0))
        self.connect((self.decoder_qpsk, 0), (self.diff_qpsk, 0))
        self.connect((self.diff_qpsk, 0), (self.unpack_qpsk, 0))
        self.connect((self.unpack_qpsk, 0), (self.corr_qpsk, 0))
        self.connect((self.corr_qpsk, 0), (self.repack_qpsk, 0))
        self.connect((self.repack_qpsk, 0), (self.crc_qpsk, 0))
        self.connect((self.crc_qpsk, 0), (self.pdu_qpsk, 0))
        self.msg_connect((self.pdu_qpsk, 'pdus'), (self.rec_qpsk, 'pdus'))

        self.connect((self.mimo_eq, 1), (self.decoder_bpsk, 0))
        self.connect((self.decoder_bpsk, 0), (self.diff_bpsk, 0))
        self.connect((self.diff_bpsk, 0), (self.corr_bpsk, 0))
        self.connect((self.corr_bpsk, 0), (self.repack_bpsk, 0))
        self.connect((self.repack_bpsk, 0), (self.crc_bpsk, 0))
        self.connect((self.crc_bpsk, 0), (self.pdu_bpsk, 0))
        self.msg_connect((self.pdu_bpsk, 'pdus'), (self.rec_bpsk, 'pdus'))

    def set_ch0_gain(self, value):
        self.gain_ch0 = value
        self.qpsk_gain_label.setText(f"<b>RX 0 Gain:</b> {self.gain_ch0} dB")
        self.uhd_usrp_source_0.set_gain(self.gain_ch0, 0)

    def set_ch1_gain(self, value):
        self.gain_ch1 = value
        self.bpsk_gain_label.setText(f"<b>RX 1 Gain:</b> {self.gain_ch1} dB")
        self.uhd_usrp_source_0.set_gain(self.gain_ch1, 1)

    def update_qpsk_image(self, path):
        pixmap = Qt.QPixmap(path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(self.qpsk_image_label.size(), Qt.Qt.KeepAspectRatio, Qt.Qt.SmoothTransformation)
            self.qpsk_image_label.setPixmap(scaled)
            self.qpsk_image_label.setStyleSheet("border: none; background: transparent;")

    def update_bpsk_image(self, path):
        pixmap = Qt.QPixmap(path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(self.bpsk_image_label.size(), Qt.Qt.KeepAspectRatio, Qt.Qt.SmoothTransformation)
            self.bpsk_image_label.setPixmap(scaled)
            self.bpsk_image_label.setStyleSheet("border: none; background: transparent;")

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