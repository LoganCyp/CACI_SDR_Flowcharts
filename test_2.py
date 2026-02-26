#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5 import Qt
from gnuradio import blocks, digital, gr, uhd
import sys
import signal

class dual_pole_tx_b210(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Dual Pole Transmitter", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Dual Pole TX (B210) - QPSK & BPSK")
        self.resize(900, 300)
        
        self.main_layout = Qt.QHBoxLayout(self)

        ##################################################
        # Variables
        ##################################################
        self.sps = 4
        self.samp_rate = 1e6
        self.excess_bw = 0.35
        self.center_freq = 2.4e9
        
        self.qpsk_access_code = '10010110110110100101000111011001'
        self.qpsk_image_path = '/home/sdr_caci1/Desktop/Test_Images/cameraman.jpg'
        self.qpsk_gain = 60

        self.bpsk_access_code = '11100001010110101110100010010011'
        self.bpsk_image_path = '/home/sdr_caci1/Desktop/Test_Images/caci-logo.jpg'
        self.bpsk_gain = 60

        ##################################################
        # GUI Setup
        ##################################################
        
        # --- Channel 0: QPSK ---
        self.qpsk_group = Qt.QGroupBox("Channel 0: QPSK Transmitter")
        self.qpsk_layout = Qt.QVBoxLayout()
        self.qpsk_group.setLayout(self.qpsk_layout)
        
        self.qpsk_gain_label = Qt.QLabel(f"<b>QPSK Gain:</b> {self.qpsk_gain} dB")
        self.qpsk_gain_slider = Qt.QSlider(Qt.Qt.Horizontal)
        self.qpsk_gain_slider.setRange(0, 90)
        self.qpsk_gain_slider.setValue(self.qpsk_gain)
        self.qpsk_gain_slider.valueChanged.connect(self.set_qpsk_gain)
        
        self.qpsk_layout.addWidget(self.qpsk_gain_label)
        self.qpsk_layout.addWidget(self.qpsk_gain_slider)
        self.main_layout.addWidget(self.qpsk_group)

        # --- Channel 1: BPSK ---
        self.bpsk_group = Qt.QGroupBox("Channel 1: BPSK Transmitter")
        self.bpsk_layout = Qt.QVBoxLayout()
        self.bpsk_group.setLayout(self.bpsk_layout)
        
        self.bpsk_gain_label = Qt.QLabel(f"<b>BPSK Gain:</b> {self.bpsk_gain} dB")
        self.bpsk_gain_slider = Qt.QSlider(Qt.Qt.Horizontal)
        self.bpsk_gain_slider.setRange(0, 90)
        self.bpsk_gain_slider.setValue(self.bpsk_gain)
        self.bpsk_gain_slider.valueChanged.connect(self.set_bpsk_gain)
        
        self.bpsk_layout.addWidget(self.bpsk_gain_label)
        self.bpsk_layout.addWidget(self.bpsk_gain_slider)
        self.main_layout.addWidget(self.bpsk_group)

        ##################################################
        # SDR Hardware Setup
        ##################################################

        # 2-Channel USRP Sink with num_send_frames testing
        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            ",".join(("", 'num_send_frames=128')),
            uhd.stream_args(cpu_format="fc32", args='', channels=list(range(0,2))),
            "",
        )
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)
        
        # Syncing the internal FPGA time for both channels
        self.uhd_usrp_sink_0.set_time_unknown_pps(uhd.time_spec())
        
        # Channel 0
        self.uhd_usrp_sink_0.set_center_freq(self.center_freq, 0)
        self.uhd_usrp_sink_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_sink_0.set_gain(self.qpsk_gain, 0)
        
        # Channel 1
        self.uhd_usrp_sink_0.set_center_freq(self.center_freq, 1)
        self.uhd_usrp_sink_0.set_antenna("TX/RX", 1)
        self.uhd_usrp_sink_0.set_gain(self.bpsk_gain, 1)

        ##################################################
        # DSP Blocks
        ##################################################

        # --- QPSK (Ch 0) ---
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_char, self.qpsk_image_path, True)
        self.blocks_stream_to_tagged_stream_0 = blocks.stream_to_tagged_stream(gr.sizeof_char, 1, 60, "packet_len")
        self.digital_protocol_formatter_bb_0 = digital.protocol_formatter_bb(
            digital.header_format_default(self.qpsk_access_code, 0), "packet_len")
        self.blocks_tagged_stream_mux_0 = blocks.tagged_stream_mux(gr.sizeof_char, "packet_len", 0)
        self.digital_constellation_modulator_0 = digital.generic_mod(
            constellation=digital.constellation_qpsk().base(),
            differential=True,
            samples_per_symbol=self.sps,
            pre_diff_code=True,
            excess_bw=self.excess_bw)

        # --- BPSK (Ch 1) ---
        self.blocks_file_source_0_0 = blocks.file_source(gr.sizeof_char, self.bpsk_image_path, True)
        self.blocks_stream_to_tagged_stream_0_0 = blocks.stream_to_tagged_stream(gr.sizeof_char, 1, 60, "packet_len")
        self.digital_protocol_formatter_bb_0_0 = digital.protocol_formatter_bb(
            digital.header_format_default(self.bpsk_access_code, 0), "packet_len")
        self.blocks_tagged_stream_mux_0_0 = blocks.tagged_stream_mux(gr.sizeof_char, "packet_len", 0)
        self.digital_constellation_modulator_0_0 = digital.generic_mod(
            constellation=digital.constellation_bpsk().base(),
            differential=True,
            samples_per_symbol=self.sps,
            pre_diff_code=True,
            excess_bw=self.excess_bw)

        ##################################################
        # Connections
        ##################################################
        
        # Ch 0
        self.connect((self.blocks_file_source_0, 0), (self.blocks_stream_to_tagged_stream_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_0, 0), (self.digital_protocol_formatter_bb_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_0, 0), (self.blocks_tagged_stream_mux_0, 1))
        self.connect((self.digital_protocol_formatter_bb_0, 0), (self.blocks_tagged_stream_mux_0, 0))
        self.connect((self.blocks_tagged_stream_mux_0, 0), (self.digital_constellation_modulator_0, 0))
        self.connect((self.digital_constellation_modulator_0, 0), (self.uhd_usrp_sink_0, 0))

        # Ch 1
        self.connect((self.blocks_file_source_0_0, 0), (self.blocks_stream_to_tagged_stream_0_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_0_0, 0), (self.digital_protocol_formatter_bb_0_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_0_0, 0), (self.blocks_tagged_stream_mux_0_0, 1))
        self.connect((self.digital_protocol_formatter_bb_0_0, 0), (self.blocks_tagged_stream_mux_0_0, 0))
        self.connect((self.blocks_tagged_stream_mux_0_0, 0), (self.digital_constellation_modulator_0_0, 0))
        self.connect((self.digital_constellation_modulator_0_0, 0), (self.uhd_usrp_sink_0, 1))

    def set_qpsk_gain(self, value):
        self.qpsk_gain = value
        self.qpsk_gain_label.setText(f"<b>QPSK Gain:</b> {value} dB")
        self.uhd_usrp_sink_0.set_gain(value, 0)

    def set_bpsk_gain(self, value):
        self.bpsk_gain = value
        self.bpsk_gain_label.setText(f"<b>BPSK Gain:</b> {value} dB")
        self.uhd_usrp_sink_0.set_gain(value, 1)

    def closeEvent(self, event):
        self.stop()
        self.wait()
        event.accept()

def main():
    qapp = Qt.QApplication(sys.argv)
    tb = dual_pole_tx_b210()
    tb.start()
    tb.show()
    signal.signal(signal.SIGINT, lambda *args: Qt.QApplication.quit())
    sys.exit(qapp.exec_())

if __name__ == '__main__':
    main()