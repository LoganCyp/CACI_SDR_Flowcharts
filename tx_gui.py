#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import blocks
import pmt
from gnuradio import digital
from gnuradio import gr
import sys
import signal
from gnuradio import uhd
import os

class RX_Test(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "SDR Image Transmitter", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("SDR Image Transmitter")
        qtgui.util.check_set_qss()
        
        self.top_layout = Qt.QVBoxLayout(self)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        ##################################################
        # Variables
        ##################################################
        self.sps = 4
        self.samp_rate = 1e6
        self.excess_bw = 0.35
        self.access_code = '11100001010110101110100010010011'
        
        # New GUI-controlled variables
        self.image_path = "/home/sdr_caci1/Desktop/Test_Images/cameraman.jpg"
        self.mod_type = "BPSK" # Default

        ##################################################
        # GUI Widgets
        ##################################################
        
        # 1. Modulation Chooser
        self._mod_type_options = ("BPSK", "QPSK")
        self._mod_type_widget = qtgui.chooser_combobox(
            label="Modulation",
            dest=self.set_mod_type,
            choices=self._mod_type_options,
            type=str,
            num_opts=len(self._mod_type_options)
        )
        self.top_grid_layout.addWidget(self._mod_type_widget, 0, 0)

        # 2. File Selection Button
        self.file_button = Qt.QPushButton("Select Image File")
        self.file_button.clicked.connect(self.open_file_dialog)
        self.top_grid_layout.addWidget(self.file_button, 0, 1)
        
        self.path_label = Qt.QLabel(f"Current File: {self.image_path}")
        self.top_grid_layout.addWidget(self.path_label, 1, 0, 1, 2)

        ##################################################
        # Blocks
        ##################################################
        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            ",".join(("", '')),
            uhd.stream_args(cpu_format="fc32", args='', channels=list(range(0,1))),
            "",
        )
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_sink_0.set_center_freq(915e6, 0)
        self.uhd_usrp_sink_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_sink_0.set_gain(50, 0)

        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_char*1, self.image_path, True)
        self.blocks_stream_to_tagged_stream_0 = blocks.stream_to_tagged_stream(gr.sizeof_char, 1, 60, "packet_len")
        self.digital_crc32_bb_0 = digital.crc32_bb(False, "packet_len", True)
        self.digital_protocol_formatter_bb_0 = digital.protocol_formatter_bb(digital.header_format_default(self.access_code, 0), "packet_len")
        self.blocks_tagged_stream_mux_0 = blocks.tagged_stream_mux(gr.sizeof_char*1, "packet_len", 0)
        
        # Initialize Modulator
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
            self.set_image_path(filename)

    def set_image_path(self, path):
        self.image_path = path
        self.path_label.setText(f"Current File: {self.image_path}")
        self.blocks_file_source_0.open(self.image_path, True)

    def set_mod_type(self, mod):
        self.mod_type = mod
        # Update the constellation object inside the existing modulator block
        if self.mod_type == "BPSK":
            new_const = digital.constellation_bpsk().base()
        else:
            new_const = digital.constellation_qpsk().base()
        
        self.digital_constellation_modulator_0.set_constellation(new_const)

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