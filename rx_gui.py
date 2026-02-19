#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import signal
import io
import pmt
import sip
from PyQt5 import Qt
from PyQt5.QtCore import pyqtSignal, QObject
from gnuradio import qtgui, analog, blocks, digital, gr, uhd, pdu

# Attempt to import PIL for image validation
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

JPEG_START = b"\xFF\xD8"
JPEG_END   = b"\xFF\xD9"

##################################################
# Helper Class for Thread-Safe UI Updates
##################################################
class SignalProxy(QObject):
    image_received = pyqtSignal(str)

##################################################
# Integrated Image Recovery Block
##################################################
class ImageRecoveryBlock(gr.basic_block):
    def __init__(self, out_jpg='recovered1.jpg'):
        gr.basic_block.__init__(self, name="Image Recovery", in_sig=None, out_sig=None)
        self.message_port_register_in(pmt.intern("pdus"))
        self.set_msg_handler(pmt.intern("pdus"), self._handle)
        
        self.out_jpg = out_jpg
        self.buf = bytearray()
        self.proxy = SignalProxy()

    def _handle(self, msg):
        # Extract data from PDU
        vec = pmt.cdr(msg)
        data = bytes(pmt.u8vector_elements(vec))
        self.buf.extend(data)

        # Search for JPEG markers
        s = self.buf.find(JPEG_START)
        if s < 0: return
        e = self.buf.find(JPEG_END, s)
        if e < 0: return

        jpg = bytes(self.buf[s:e+2])

        # Validate and save if correct
        if self._valid_jpeg(jpg):
            with open(self.out_jpg, "wb") as f:
                f.write(jpg)
            # Notify the GUI thread
            self.proxy.image_received.emit(self.out_jpg)
            # Clear processed data from buffer
            self.buf = self.buf[e+2:]
        else:
            # Shift buffer past the false marker to continue searching
            del self.buf[:s+2]

    def _valid_jpeg(self, b):
        if len(b) < 100: return False
        if not HAS_PIL: return True # Fallback if Pillow isn't installed
        try:
            im = Image.open(io.BytesIO(b))
            im.verify()
            return True
        except:
            return False

##################################################
# Main GUI and Flowgraph Class
##################################################
class caci_rtl_rx(gr.top_block, Qt.QWidget):
    def __init__(self):
        gr.top_block.__init__(self, "SDR Image Receiver", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("SDR Image Receiver - Continuous Mode")
        self.resize(1100, 600)

        # Layout
        self.top_layout = Qt.QVBoxLayout(self)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        # Variables
        self.sps = 4
        self.samp_rate = 1e6
        self.freq = 2.4e9  # Matches the BPSK transmitter
        self.access_code = '11100001010110101110100010010011'

        # GUI Widgets
        self.image_label = Qt.QLabel("Awaiting Image...")
        self.image_label.setAlignment(Qt.Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            font-size: 24px; 
            color: #555; 
            border: 3px dashed #bbb; 
            background: #f9f9f9;
            border-radius: 10px;
        """)
        self.top_grid_layout.addWidget(self.image_label, 0, 0)

        self.qtgui_const_sink_x_0 = qtgui.const_sink_c(1024, "Receiver Constellation", 1)
        self.qtgui_const_sink_x_0.set_update_time(0.10)
        self._const_win = sip.wrapinstance(self.qtgui_const_sink_x_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(self._const_win, 0, 1)

        ##################################################
        # Blocks
        ##################################################
        self.uhd_source = uhd.usrp_source(
            ",".join(("", '')),
            uhd.stream_args(cpu_format="fc32", args='', channels=list(range(0,1))),
        )
        self.uhd_source.set_samp_rate(self.samp_rate)
        self.uhd_source.set_center_freq(self.freq, 0)
        self.uhd_source.set_gain(50, 0)

        self.agc = analog.agc_cc(1e-4, 1.0, 1.0, 4000)
        self.fll = digital.fll_band_edge_cc(self.sps, 0.35, 45, 0.05)
        self.clock_sync = digital.pfb_clock_sync_ccf(self.sps, 0.0628, 
            digital.firdes.root_raised_cosine(32, 32, 1.0, 0.35, 11*32), 32, 16, 1.5, 1)
        self.costas = digital.costas_loop_cc(0.05, 2, False)
        self.decoder = digital.constellation_decoder_cb(digital.constellation_bpsk().base())
        self.diff_dec = digital.diff_decoder_bb(2, digital.DIFF_DIFFERENTIAL)
        self.corr = digital.correlate_access_code_bb_ts(self.access_code, 1, "packet_len")
        self.repack = blocks.repack_bits_bb(1, 8, "packet_len", True, gr.GR_MSB_FIRST)
        self.crc = digital.crc32_bb(True, "packet_len", True)
        self.ts_to_pdu = pdu.tagged_stream_to_pdu(gr.types.byte_t, 'packet_len')
        
        # Instantiate our integrated recovery block
        self.recovery_block = ImageRecoveryBlock(out_jpg='recovered_latest.jpg')

        # Connect the signal for thread-safe UI updates
        self.recovery_block.proxy.image_received.connect(self.update_image_display)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.uhd_source, 0), (self.agc, 0))
        self.connect((self.agc, 0), (self.fll, 0))
        self.connect((self.fll, 0), (self.clock_sync, 0))
        self.connect((self.clock_sync, 0), (self.costas, 0))
        self.connect((self.costas, 0), (self.qtgui_const_sink_x_0, 0))
        self.connect((self.costas, 0), (self.decoder, 0))
        self.connect((self.decoder, 0), (self.diff_dec, 0))
        self.connect((self.diff_dec, 0), (self.corr, 0))
        self.connect((self.corr, 0), (self.repack, 0))
        self.connect((self.repack, 0), (self.crc, 0))
        self.connect((self.crc, 0), (self.ts_to_pdu, 0))
        self.msg_connect((self.ts_to_pdu, 'pdus'), (self.recovery_block, 'pdus'))

    def update_image_display(self, path):
        pixmap = Qt.QPixmap(path)
        if not pixmap.isNull():
            # Scale to fit while maintaining quality
            scaled = pixmap.scaled(self.image_label.size(), 
                                   Qt.Qt.KeepAspectRatio, 
                                   Qt.Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled)
            self.image_label.setStyleSheet("border: none; background: transparent;")

    def closeEvent(self, event):
        self.stop()
        self.wait()
        event.accept()

def main():
    qapp = Qt.QApplication(sys.argv)
    tb = caci_rtl_rx()
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