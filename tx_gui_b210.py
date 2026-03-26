#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5 import Qt
from gnuradio import qtgui, blocks, digital, gr, uhd
import pmt
import sys
import signal

class dual_pole_tx_b210(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Dual Pole Transmitter", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Dual Pole TX (B210) - BPSK & BPSK")
        self.resize(900, 300)
        
        # Main Layout: Side-by-side columns
        self.main_layout = Qt.QHBoxLayout(self)

        ##################################################
        # Variables
        ##################################################
        self.sps = 4
        self.samp_rate = 1e6
        self.excess_bw = 0.35
        self.center_freq = 2.4e9
        
        # Channel 0 (BPSK) Variables
        self.bpsk0_access_code = '10010110110110100101000111011001'
        self.bpsk0_image_path = '/home/sdr_caci1/Desktop/Test_Images/cameraman.jpg'
        self.bpsk0_gain = 50

        # Channel 1 (BPSK) Variables
        self.bpsk1_access_code = '11100001010110101110100010010011'
        self.bpsk1_image_path = '/home/sdr_caci1/Desktop/Test_Images/caci-logo.jpg'
        self.bpsk1_gain = 50

        ##################################################
        # GUI Setup
        ##################################################
        
        # --- BPSK Control Group (Channel 0) ---
        self.bpsk0_group = Qt.QGroupBox("Channel 0: BPSK Transmitter")
        self.bpsk0_layout = Qt.QVBoxLayout()
        self.bpsk0_group.setLayout(self.bpsk0_layout)
        
        self.bpsk0_file_btn = Qt.QPushButton("Select BPSK Ch0 Image...")
        self.bpsk0_file_btn.clicked.connect(self.open_bpsk0_file_dialog)
        self.bpsk0_layout.addWidget(self.bpsk0_file_btn)
        
        self.bpsk0_path_display = Qt.QLineEdit(self.bpsk0_image_path)
        self.bpsk0_path_display.setReadOnly(True)
        self.bpsk0_layout.addWidget(self.bpsk0_path_display)
        
        self.bpsk0_gain_label = Qt.QLabel(f"<b>BPSK Ch0 Gain:</b> {self.bpsk0_gain} dB")
        self.bpsk0_layout.addWidget(self.bpsk0_gain_label)
        
        self.bpsk0_gain_slider = Qt.QSlider(Qt.Qt.Horizontal)
        self.bpsk0_gain_slider.setRange(0, 90)
        self.bpsk0_gain_slider.setValue(self.bpsk0_gain)
        self.bpsk0_gain_slider.valueChanged.connect(self.set_bpsk0_gain)
        self.bpsk0_layout.addWidget(self.bpsk0_gain_slider)
        
        self.main_layout.addWidget(self.bpsk0_group)

        # --- BPSK Control Group (Channel 1) ---
        self.bpsk1_group = Qt.QGroupBox("Channel 1: BPSK Transmitter")
        self.bpsk1_layout = Qt.QVBoxLayout()
        self.bpsk1_group.setLayout(self.bpsk1_layout)
        
        self.bpsk1_file_btn = Qt.QPushButton("Select BPSK Ch1 Image...")
        self.bpsk1_file_btn.clicked.connect(self.open_bpsk1_file_dialog)
        self.bpsk1_layout.addWidget(self.bpsk1_file_btn)
        
        self.bpsk1_path_display = Qt.QLineEdit(self.bpsk1_image_path)
        self.bpsk1_path_display.setReadOnly(True)
        self.bpsk1_layout.addWidget(self.bpsk1_path_display)
        
        self.bpsk1_gain_label = Qt.QLabel(f"<b>BPSK Ch1 Gain:</b> {self.bpsk1_gain} dB")
        self.bpsk1_layout.addWidget(self.bpsk1_gain_label)
        
        self.bpsk1_gain_slider = Qt.QSlider(Qt.Qt.Horizontal)
        self.bpsk1_gain_slider.setRange(0, 90)
        self.bpsk1_gain_slider.setValue(self.bpsk1_gain)
        self.bpsk1_gain_slider.valueChanged.connect(self.set_bpsk1_gain)
        self.bpsk1_layout.addWidget(self.bpsk1_gain_slider)
        
        self.main_layout.addWidget(self.bpsk1_group)

        ##################################################
        # Blocks setup
        ##################################################

        # 2-Channel USRP Sink
        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            ",".join(("", '')),
            uhd.stream_args(cpu_format="fc32", args='', channels=list(range(0,2))),
            "",
        )
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)
        
        # Channel 0 Init
        self.uhd_usrp_sink_0.set_center_freq(self.center_freq, 0)
        self.uhd_usrp_sink_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_sink_0.set_gain(self.bpsk0_gain, 0)
        
        # Channel 1 Init
        self.uhd_usrp_sink_0.set_center_freq(self.center_freq, 1)
        self.uhd_usrp_sink_0.set_antenna("TX/RX", 1)
        self.uhd_usrp_sink_0.set_gain(self.bpsk1_gain, 1)

        # --- BPSK Blocks (Channel 0) ---
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_char, self.bpsk0_image_path, True)
        self.blocks_stream_to_tagged_stream_0 = blocks.stream_to_tagged_stream(gr.sizeof_char, 1, 60, "packet_len")
        self.digital_crc32_bb_0 = digital.crc32_bb(False, "packet_len", True)
        self.digital_protocol_formatter_bb_0 = digital.protocol_formatter_bb(
            digital.header_format_default(self.bpsk0_access_code, 0), "packet_len")
        self.blocks_tagged_stream_mux_0 = blocks.tagged_stream_mux(gr.sizeof_char, "packet_len", 0)
        self.digital_constellation_modulator_0 = digital.generic_mod(
            constellation=digital.constellation_bpsk().base(),
            differential=True,
            samples_per_symbol=self.sps,
            pre_diff_code=True,
            excess_bw=self.excess_bw)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_cc(0.7)

        # --- BPSK Blocks (Channel 1) ---
        self.blocks_file_source_0_0 = blocks.file_source(gr.sizeof_char, self.bpsk1_image_path, True)
        self.blocks_stream_to_tagged_stream_0_0 = blocks.stream_to_tagged_stream(gr.sizeof_char, 1, 60, "packet_len")
        self.digital_crc32_bb_0_0 = digital.crc32_bb(False, "packet_len", True)
        self.digital_protocol_formatter_bb_0_0 = digital.protocol_formatter_bb(
            digital.header_format_default(self.bpsk1_access_code, 0), "packet_len")
        self.blocks_tagged_stream_mux_0_0 = blocks.tagged_stream_mux(gr.sizeof_char, "packet_len", 0)
        self.digital_constellation_modulator_0_0 = digital.generic_mod(
            constellation=digital.constellation_bpsk().base(),
            differential=True,
            samples_per_symbol=self.sps,
            pre_diff_code=True,
            excess_bw=self.excess_bw)
        self.blocks_multiply_const_vxx_0_0 = blocks.multiply_const_cc(0.7)


        ##################################################
        # Connections
        ##################################################
        
        # BPSK Path Connections -> USRP Channel 0
        self.connect((self.blocks_file_source_0, 0), (self.blocks_stream_to_tagged_stream_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_0, 0), (self.digital_crc32_bb_0, 0))
        self.connect((self.digital_crc32_bb_0, 0), (self.digital_protocol_formatter_bb_0, 0))
        self.connect((self.digital_crc32_bb_0, 0), (self.blocks_tagged_stream_mux_0, 1))
        self.connect((self.digital_protocol_formatter_bb_0, 0), (self.blocks_tagged_stream_mux_0, 0))
        self.connect((self.blocks_tagged_stream_mux_0, 0), (self.digital_constellation_modulator_0, 0))
        self.connect((self.digital_constellation_modulator_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.uhd_usrp_sink_0, 0))

        # BPSK Path Connections -> USRP Channel 1
        self.connect((self.blocks_file_source_0_0, 0), (self.blocks_stream_to_tagged_stream_0_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_0_0, 0), (self.digital_crc32_bb_0_0, 0))
        self.connect((self.digital_crc32_bb_0_0, 0), (self.digital_protocol_formatter_bb_0_0, 0))
        self.connect((self.digital_crc32_bb_0_0, 0), (self.blocks_tagged_stream_mux_0_0, 1))
        self.connect((self.digital_protocol_formatter_bb_0_0, 0), (self.blocks_tagged_stream_mux_0_0, 0))
        self.connect((self.blocks_tagged_stream_mux_0_0, 0), (self.digital_constellation_modulator_0_0, 0))
        self.connect((self.digital_constellation_modulator_0_0, 0), (self.blocks_multiply_const_vxx_0_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0_0, 0), (self.uhd_usrp_sink_0, 1))

    ##################################################
    # Callbacks
    ##################################################

    def open_bpsk0_file_dialog(self):
        filename, _ = Qt.QFileDialog.getOpenFileName(self, "Open BPSK Ch0 Image", "", "Images (*.jpg *.png *.jpeg);;All Files (*)")
        if filename:
            self.bpsk0_image_path = filename
            self.bpsk0_path_display.setText(filename)
            self.blocks_file_source_0.open(self.bpsk0_image_path, True)

    def open_bpsk1_file_dialog(self):
        filename, _ = Qt.QFileDialog.getOpenFileName(self, "Open BPSK Ch1 Image", "", "Images (*.jpg *.png *.jpeg);;All Files (*)")
        if filename:
            self.bpsk1_image_path = filename
            self.bpsk1_path_display.setText(filename)
            self.blocks_file_source_0_0.open(self.bpsk1_image_path, True)

    def set_bpsk0_gain(self, value):
        self.bpsk0_gain = value
        self.bpsk0_gain_label.setText(f"<b>BPSK Ch0 Gain:</b> {self.bpsk0_gain} dB")
        self.uhd_usrp_sink_0.set_gain(self.bpsk0_gain, 0)

    def set_bpsk1_gain(self, value):
        self.bpsk1_gain = value
        self.bpsk1_gain_label.setText(f"<b>BPSK Ch1 Gain:</b> {self.bpsk1_gain} dB")
        self.uhd_usrp_sink_0.set_gain(self.bpsk1_gain, 1)

    def closeEvent(self, event):
        self.stop()
        self.wait()
        event.accept()

def main():
    qapp = Qt.QApplication(sys.argv)
    tb = dual_pole_tx_b210()
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