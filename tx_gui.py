#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5 import Qt
from gnuradio import qtgui, blocks, digital, gr, uhd
import pmt
import sys
import signal

class RX_Test(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "BPSK Image Transmitter", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("BPSK Image Transmitter")
        
        # Set window size
        self.resize(700, 300)
        
        # Main Layout
        self.layout = Qt.QVBoxLayout(self)
        self.grid = Qt.QGridLayout()
        self.layout.addLayout(self.grid)

        ##################################################
        # Parameters
        ##################################################
        self.sps = 4
        self.samp_rate = 1e6
        self.excess_bw = 0.35
        self.center_freq = 2.4e9  # Updated to 2.4 GHz
        self.gain = 50            # Initial Gain
        self.access_code = '11100001010110101110100010010011'
        self.image_path = "/home/sdr_caci1/Desktop/Sample Images/cameraman.jpg"

        ##################################################
        # GUI Widgets
        ##################################################
        
        # 1. File Selection Row
        self.file_button = Qt.QPushButton("Select Image...")
        self.file_button.clicked.connect(self.open_file_dialog)
        self.grid.addWidget(self.file_button, 0, 0)
        
        self.path_display = Qt.QLineEdit(self.image_path)
        self.path_display.setReadOnly(True)
        self.grid.addWidget(self.path_display, 0, 1)

        # 2. Gain Slider Row
        self.gain_label = Qt.QLabel(f"<b>USRP Gain:</b> {self.gain} dB")
        self.grid.addWidget(self.gain_label, 1, 0)
        
        self.gain_slider = Qt.QSlider(Qt.Qt.Horizontal)
        self.gain_slider.setRange(0, 90) # Standard range for many USRPs
        self.gain_slider.setValue(int(self.gain))
        self.gain_slider.valueChanged.connect(self.set_usrp_gain)
        self.grid.addWidget(self.gain_slider, 1, 1)

        # 3. Frequency Info Row
        self.freq_info = Qt.QLabel(f"<b>Frequency:</b> {self.center_freq/1e9} GHz")
        self.grid.addWidget(self.freq_info, 2, 0, 1, 2)

        ##################################################
        # Blocks
        ##################################################
        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            ",".join(("", '')),
            uhd.stream_args(cpu_format="fc32", args='', channels=list(range(0,1))),
            "",
        )
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_sink_0.set_center_freq(self.center_freq, 0)
        self.uhd_usrp_sink_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_sink_0.set_gain(self.gain, 0)

        # File Source
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_char, self.image_path, True)
        self.blocks_stream_to_tagged_stream_0 = blocks.stream_to_tagged_stream(gr.sizeof_char, 1, 60, "packet_len")
        self.digital_crc32_bb_0 = digital.crc32_bb(False, "packet_len", True)
        self.digital_protocol_formatter_bb_0 = digital.protocol_formatter_bb(
            digital.header_format_default(self.access_code, 0), "packet_len")
        self.blocks_tagged_stream_mux_0 = blocks.tagged_stream_mux(gr.sizeof_char, "packet_len", 0)
        
        # BPSK Modulator
        self.digital_constellation_modulator_0 = digital.generic_mod(
            constellation=digital.constellation_bpsk().base(),
            differential=True,
            samples_per_symbol=self.sps,
            pre_diff_code=True,
            excess_bw=self.excess_bw)
            
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_cc(0.7)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_file_source_0, 0), (self.blocks_stream_to_tagged_stream_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_0, 0), (self.digital_crc32_bb_0, 0))
        self.connect((self.digital_crc32_bb_0, 0), (self.digital_protocol_formatter_bb_0, 0))
        self.connect((self.digital_crc32_bb_0, 0), (self.blocks_tagged_stream_mux_0, 1))
        self.connect((self.digital_protocol_formatter_bb_0, 0), (self.blocks_tagged_stream_mux_0, 0))
        self.connect((self.blocks_tagged_stream_mux_0, 0), (self.digital_constellation_modulator_0, 0))
        self.connect((self.digital_constellation_modulator_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.uhd_usrp_sink_0, 0))

    def open_file_dialog(self):
        filename, _ = Qt.QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.jpg *.png *.jpeg);;All Files (*)")
        if filename:
            self.image_path = filename
            self.path_display.setText(filename)
            self.blocks_file_source_0.open(self.image_path, True)

    def set_usrp_gain(self, value):
        self.gain = value
        self.gain_label.setText(f"<b>USRP Gain:</b> {self.gain} dB")
        self.uhd_usrp_sink_0.set_gain(self.gain, 0)

    def closeEvent(self, event):
        self.stop()
        self.wait()
        event.accept()

def main():
    qapp = Qt.QApplication(sys.argv)
    tb = RX_Test()
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