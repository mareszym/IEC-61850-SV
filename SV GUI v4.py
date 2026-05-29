#!/usr/bin/env python3

import sys
import time
import struct
import numpy as np
import os
import psutil
import xml.etree.ElementTree as ET
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox,
    QFormLayout, QTabWidget, QTextEdit, QGroupBox, QMessageBox
)
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot, Qt
import pyqtgraph as pg

# from scapy.all import Ether, Dot1Q, Raw, sendp, sniff, get_if_list, get_if_hwaddr

from scapy.layers.l2 import Ether, Dot1Q
from scapy.packet import Raw
from scapy.sendrecv import sendp, sniff
from scapy.all import get_if_list, get_if_hwaddr

import scapy.arch


# ==============================================================================
#  1. SV PUBLISHER LOGIC (Manual ASN.1 BER Encoding)
# ==============================================================================

class SVWorker(QObject):
    """
    Worker thread for generating and transmitting SV packets at high frequency.
    """
    new_data_signal = pyqtSignal(np.ndarray)
    packets_sent_signal = pyqtSignal(int)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._is_running = True
        self.packet_count = 0

    @staticmethod
    def _create_tlv(tag, value_bytes):
        """Creates a simple ASN.1 Type-Length-Value structure."""
        tag_byte = bytes([tag])
        # This assumes length is less than 128, which is true for these fields
        len_byte = bytes([len(value_bytes)])
        return tag_byte + len_byte + value_bytes

    def _encode_apdu(self, svID, smpCnt, confRev, smpSynch, data_set):
        """
        Encodes the APDU payload using manual ASN.1 BER construction.
        """
        # 1. Create TLVs for each field inside the ASDU
        svid_tlv = self._create_tlv(0x80, svID.encode('ascii'))
        smpcnt_tlv = self._create_tlv(0x82, smpCnt.to_bytes(2, 'big'))
        confrev_tlv = self._create_tlv(0x83, confRev.to_bytes(4, 'big'))
        smpsynch_tlv = self._create_tlv(0x85, smpSynch.to_bytes(1, 'big'))
        dataset_tlv = self._create_tlv(0x87, data_set)

        # 2. Concatenate the field TLVs to form the content of the ASDU
        asdu_content = svid_tlv + smpcnt_tlv + confrev_tlv + smpsynch_tlv + dataset_tlv
        
        # 3. Wrap the content in an ASDU SEQUENCE tag (0x30)
        asdu_tlv = self._create_tlv(0x30, asdu_content)

        # 4. Create the 'seqASDU' (a constructed type with tag A2)
        seq_asdu_tlv = self._create_tlv(0xA2, asdu_tlv)

        # 5. Create the 'noASDU' TLV
        no_asdu_tlv = self._create_tlv(0x80, (1).to_bytes(1, 'big'))

        # 6. Concatenate to form the content of the savPDU
        sav_pdu_content = no_asdu_tlv + seq_asdu_tlv
        
        # 7. Wrap in the final savPDU application tag (0x60)
        sav_pdu_tlv = self._create_tlv(0x60, sav_pdu_content)
        
        return sav_pdu_tlv

    @pyqtSlot()
    def run(self):
        freq = self.config['freq']
        sps = self.config['sps']
        sampling_rate = freq * sps
        time_step = 1.0 / sampling_rate

        smpCnt = 0
        confRev = 1
        smpSynch = 2  # Global sync (e.g., PTP)

        vlan_layer = Dot1Q(vlan=self.config['vlan_id'], prio=4) if self.config['vlan_id'] > 0 else None

        # Pre-generate one full cycle of waveform data for efficiency

        w = 2 * np.pi * freq
        v_amp = self.config['v_amp']
        i_amp = self.config['i_amp']
        v3h_amp = self.config['v3H_amp_in']/100
        v5h_amp = self.config['v5H_amp_in']/100
        cos_phi = self.config['cos_phi']
        phi = np.arccos(cos_phi)  # [rad]

        t = np.arange(0, sps) * time_step
        # va = self.config['v_amp'] * np.sin(2 * np.pi * freq * t)
        # vb = self.config['v_amp'] * np.sin(2 * np.pi * freq * t - 2 * np.pi / 3)
        # vc = self.config['v_amp'] * np.sin(2 * np.pi * freq * t + 2 * np.pi / 3)
        # ia = self.config['i_amp'] * np.sin(2 * np.pi * freq * t)
        # ib = self.config['i_amp'] * np.sin(2 * np.pi * freq * t - 2 * np.pi / 3)
        # ic = self.config['i_amp'] * np.sin(2 * np.pi * freq * t + 2 * np.pi / 3)

        # --- VOLTAGE ---
        va = (
                v_amp * np.sin(w * t) +
                v3h_amp * v_amp * np.sin(3 * w * t) +
                v5h_amp * v_amp * np.sin(5 * w * t)
        )

        vb = (
                v_amp * np.sin(w * t - 2 * np.pi / 3) +
                v3h_amp * v_amp * np.sin(3 * w * t) +
                v5h_amp * v_amp * np.sin(5 * w * t - 10 * np.pi / 3)
        )

        vc = (
                v_amp * np.sin(w * t + 2 * np.pi / 3) +
                v3h_amp * v_amp * np.sin(3 * w * t) +
                v5h_amp * v_amp * np.sin(5 * w * t + 10 * np.pi / 3)
        )

        # # --- CURRENT ---
        # ia = (
        #         i_amp * np.sin(w * t) +
        #         v3h_amp * i_amp * np.sin(3 * w * t) +
        #         v5h_amp * i_amp * np.sin(5 * w * t)
        # )
        #
        # ib = (
        #         i_amp * np.sin(w * t - 2 * np.pi / 3) +
        #         v3h_amp * i_amp * np.sin(3 * w * t) +
        #         v5h_amp * i_amp * np.sin(5 * w * t - 10 * np.pi / 3)
        # )
        #
        # ic = (
        #         i_amp * np.sin(w * t + 2 * np.pi / 3) +
        #         v3h_amp * i_amp * np.sin(3 * w * t) +
        #         v5h_amp * i_amp * np.sin(5 * w * t + 10 * np.pi / 3)
        # )

        # --- CURRENT with cos(phi) ---
        ia = (
                i_amp * np.sin(w * t - phi) +
                v3h_amp * i_amp * np.sin(3 * w * t) +
                v5h_amp * i_amp * np.sin(5 * w * t)
        )

        ib = (
                i_amp * np.sin(w * t - 2 * np.pi / 3 - phi) +
                v3h_amp * i_amp * np.sin(3 * w * t) +
                v5h_amp * i_amp * np.sin(5 * w * t - 10 * np.pi / 3)
        )

        ic = (
                i_amp * np.sin(w * t + 2 * np.pi / 3 - phi) +
                v3h_amp * i_amp * np.sin(3 * w * t) +
                v5h_amp * i_amp * np.sin(5 * w * t + 10 * np.pi / 3)
        )

        gui_buffer = []
        
        while self._is_running:
            start_time = time.perf_counter()
            idx = smpCnt % sps

            # According to 9-2LE: Currents in mA, Voltages in 10mV
            data_values = [
                int(ia[idx] * 1000), 0, int(ib[idx] * 1000), 0,
                int(ic[idx] * 1000), 0, 0, 0,  # IN, INq
                int(va[idx] * 100), 0, int(vb[idx] * 100), 0,
                int(vc[idx] * 100), 0, 0, 0,  # VN, VNq
            ]
            data_set_bytes = struct.pack('!16i', *data_values)

            apdu_payload = self._encode_apdu(self.config['sv_id'], smpCnt, confRev, smpSynch, data_set_bytes)

            # SV Header: APPID, Length, Reserved1 (Simulate bit set), Reserved2
            sv_header = struct.pack('>HHHH', int(self.config['appid'], 16), len(apdu_payload) + 8, 0x8000, 0)
            
            ether_layer = Ether(src=self.config['src_mac'], dst=self.config['dst_mac'], type=0x88BA)
            
            if vlan_layer:
                packet = ether_layer / vlan_layer / Raw(load=sv_header + apdu_payload)
            else:
                packet = ether_layer / Raw(load=sv_header + apdu_payload)

            sendp(packet, iface=self.config['iface'], verbose=0)

            self.packet_count += 1
            smpCnt = (smpCnt + 1) % sampling_rate

            gui_buffer.append(data_values)
            if len(gui_buffer) >= 100: # Update GUI every 100 samples
                self.new_data_signal.emit(np.array(gui_buffer))
                self.packets_sent_signal.emit(self.packet_count)
                gui_buffer = []

            # Precise sleep to maintain sampling rate
            elapsed = time.perf_counter() - start_time
            sleep_time = time_step - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # FIX: Even if we're behind schedule, yield control to the OS 
                # to prevent the GUI from freezing. A zero-duration sleep
                # is usually sufficient to force a context switch.
                time.sleep(0)

    def stop(self):
        self._is_running = False

# ==============================================================================
#  2. SV DECODER LOGIC (Manual ASN.1 BER Parsing)
# ==============================================================================

class DecoderWorker(QObject):
    packet_decoded = pyqtSignal(str)

    def __init__(self, iface):
        super().__init__()
        self.iface = iface
        self.running = False

    @staticmethod
    def _parse_tlv(data):
        """Parses a stream of TLVs and returns a dictionary."""
        decoded = {}
        i = 0
        while i < len(data):
            tag = data[i]
            length = data[i+1]
            value_bytes = data[i+2 : i+2+length]
            
            if tag == 0x80: decoded['svID'] = value_bytes.decode('ascii')
            elif tag == 0x82: decoded['smpCnt'] = int.from_bytes(value_bytes, 'big')
            elif tag == 0x83: decoded['confRev'] = int.from_bytes(value_bytes, 'big')
            elif tag == 0x85: decoded['smpSynch'] = int.from_bytes(value_bytes, 'big')
            elif tag == 0x87:
                d = struct.unpack('!16i', value_bytes)
                decoded['DataSet'] = {
                    'Ia': d[0]/1000.0, 'Iaq': hex(d[1]), 'Ib': d[2]/1000.0, 'Ibq': hex(d[3]),
                    'Ic': d[4]/1000.0, 'Icq': hex(d[5]), 'In': d[6]/1000.0, 'Inq': hex(d[7]),
                    'Va': d[8]/100.0, 'Vaq': hex(d[9]), 'Vb': d[10]/100.0, 'Vbq': hex(d[11]),
                    'Vc': d[12]/100.0, 'Vcq': hex(d[13]), 'Vn': d[14]/100.0, 'Vnq': hex(d[15]),
                }
            i += 2 + length
        return decoded

    def _decode_sv_packet(self, packet):
        if Raw not in packet: return None
        payload = packet.load
        
        appid, length, res1, res2 = struct.unpack('>HHHH', payload[:8])
        is_simulated = (res1 & 0x8000) != 0
        
        apdu_data = payload[8:]
        
        # Manually parse the APDU
        sav_pdu_tag = apdu_data[0]
        if sav_pdu_tag != 0x60: return "Not a savPDU"
        
        sav_pdu_len = apdu_data[1]
        sav_pdu_content = apdu_data[2 : 2+sav_pdu_len]
        
        decoded = {'APPID': hex(appid), 'Simulated': is_simulated}
        
        # no_asdu_tag = sav_pdu_content[0]
        no_asdu_len = sav_pdu_content[1]
        no_asdu_val = int.from_bytes(sav_pdu_content[2:2+no_asdu_len], 'big')
        decoded['noASDU'] = no_asdu_val
        
        offset = 2 + no_asdu_len
        seq_asdu_tag = sav_pdu_content[offset]
        if seq_asdu_tag != 0xA2: return "seqASDU not found"
        
        seq_asdu_len = sav_pdu_content[offset+1]
        seq_asdu_content = sav_pdu_content[offset+2 : offset+2+seq_asdu_len]
        
        asdu_tag = seq_asdu_content[0]
        if asdu_tag != 0x30: return "ASDU not found"
        
        asdu_len = seq_asdu_content[1]
        asdu_content = seq_asdu_content[2 : 2+asdu_len]
        
        # Parse the fields inside the ASDU
        asdu_fields = self._parse_tlv(asdu_content)
        decoded.update(asdu_fields)
        
        return decoded

    def _process_packet(self, packet):
        decoded_data = self._decode_sv_packet(packet)
        if isinstance(decoded_data, dict):
            output = f"--- SV Packet ---\n"
            output += f"  Source MAC: {packet[Ether].src}\n"
            for key, value in decoded_data.items():
                if key == 'DataSet' and isinstance(value, dict): #jawne sprawdzenie zmiennej value - typ słownikowy
                    output += f"  {key}:\n"
                    for k, v in value.items():
                        output += f"    {k:<4}: {v:.4f}" if isinstance(v, float) else f"    {k:<4}: {v}"
                        output += "\n"
                else:
                    output += f"  {key:<12}: {value}\n"
            self.packet_decoded.emit(output)

    @pyqtSlot()
    def run(self):
        self.running = True
        # Loop with a timeout to prevent sniff from blocking indefinitely
        while self.running:
            sniff(iface=self.iface, filter="ether proto 0x88ba", prn=self._process_packet, stop_filter=lambda p: not self.running, timeout=1)

    def stop(self):
        self.running = False

# ==============================================================================
#  3. MAIN GUI APPLICATION
# ==============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.plot_data_v = None
        self.plot_data_i = None
        self.setWindowTitle("IEC 61850-9-2 SV Simulator")
        self.setGeometry(100, 100, 1200, 800)

        # --- THEME AND STYLING ---
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #f0f0f0;
                color: #000000; /* Default font color to black */
            }
            QGroupBox {
                background-color: #e0e8f0;
                border: 1px solid #c0c0c0;
                border-radius: 5px;
                margin-top: 1ex; 
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
                color: #000000;
            }
            QLabel {
                color: #000000;
                font-weight: bold;
            }
            QLineEdit, QTextEdit, QComboBox {
                background-color: #ffffff;
                border: 1px solid #c0c0c0;
                border-radius: 4px;
                padding: 4px;
                color: #000000;
            }
            
            QSpinBox, QDoubleSpinBox {
                background-color: #ffffff;
                border: 1px solid #c0c0c0;
                border-radius: 4px;
                padding-right: 20px;   /* miejsce na przyciski */
                color: #000000;
            }
            
            /* Przyciski ▲▼ */
            QSpinBox::up-button, QDoubleSpinBox::up-button,
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                width: 22px;
                height: 12px;
            }

            QPushButton {
                background-color: #0078d7;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #a0a0a0;
            }
            QTabWidget::pane {
                border-top: 1px solid #c0c0c0;
            }
            QTabBar::tab {
                background: #d0d0d0;
                border: 1px solid #c0c0c0;
                padding: 6px;
                border-bottom-left-radius: 4px;
                border-bottom-right-radius: 4px;
                color: #000000;
            }
            QTabBar::tab:selected {
                background: #e0e8f0;
                margin-bottom: -1px; 
            }
        """)

        self.worker = None
        self.thread = None
        self.decoder_worker = None
        self.decoder_thread = None

        # Create a main widget and layout that will hold the tabs and the credit label
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 5)
        main_layout.setSpacing(10)

        self.tabs = QTabWidget()
        self._create_publisher_tab()
        self._create_decoder_tab()
        
        main_layout.addWidget(self.tabs)

        # --- CREDIT LABEL INSERTION ---
        credit_label = QLabel("<b>Developed by Sugandh Pratap, Powered By Mareszym - 2026</b>")
        credit_label.setStyleSheet("font-size: 12pt; color: #333333;")
        credit_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(credit_label)
        # --- END OF INSERTION ---

        self.setCentralWidget(main_widget)

    def _create_publisher_tab(self):
        tab = QWidget()
        main_layout = QHBoxLayout(tab)
        
        # --- Controls GroupBox ---
        controls_group = QGroupBox("Simulation Configuration")
        controls_layout = QVBoxLayout()
        main_layout.addWidget(controls_group, 1)
        
        form_layout = QFormLayout()
        # available_ifaces = get_if_list()
        # self.iface_in = QComboBox()
        # self.iface_in.addItems(available_ifaces)
        # self.src_mac_in = QLineEdit(get_if_hwaddr(self.iface_in.currentText()))
        # self.iface_in.currentTextChanged.connect(lambda iface: self.src_mac_in.setText(get_if_hwaddr(iface)))

        # self.iface_in = QComboBox()
        #
        # for iface in get_windows_if_list():
        #     name = iface.get("name", "Unknown")
        #     desc = iface.get("description", "")
        #     guid = iface.get("guid")
        #
        #     if not guid:
        #         continue
        #
        #     # ✅ KLUCZOWE: z GUID robimy nazwę NPF
        #     npf = rf"\Device\NPF_{guid}"
        #
        #     display = f"{name} – {desc}"
        #     self.iface_in.addItem(display, npf)
        #
        # # ✅ Source MAC – POPRAWNIE
        # current_npf = self.iface_in.currentData()
        # self.src_mac_in = QLineEdit(get_if_hwaddr(current_npf))
        #
        # self.iface_in.currentIndexChanged.connect(
        #     lambda _: self.src_mac_in.setText(
        #         get_if_hwaddr(self.iface_in.currentData())
        #     )
        # )

        self.iface_in = QComboBox()

        for display, npf in self._get_active_windows_interfaces:
            self.iface_in.addItem(display, npf)

        # --- Source MAC ---
        current_npf = self.iface_in.currentData()
        if current_npf:
            self.src_mac_in = QLineEdit(get_if_hwaddr(current_npf))
        else:
            self.src_mac_in = QLineEdit("00:00:00:00:00:00")

        self.iface_in.currentIndexChanged.connect(
            lambda _: self.src_mac_in.setText(
                get_if_hwaddr(self.iface_in.currentData())
            )
        )

        self.dst_mac_in = QLineEdit("01:0C:CD:04:00:01")
        self.vlan_id_in = QSpinBox()
        self.vlan_id_in.setMinimum(0)
        self.vlan_id_in.setMaximum(4095)
        self.vlan_id_in.setValue(0)
        self.appid_in = QLineEdit("4000")
        self.svid_in = QLineEdit("SimulatedSVStream")
        self.freq_in = QComboBox()
        self.freq_in.addItems(["50", "60"])
        self.freq_in.setCurrentIndex(0)
        self.sps_in = QComboBox()
        self.sps_in.addItems(["80", "256"])
        self.sps_in.setCurrentIndex(1)

        self.v_amp_in = QDoubleSpinBox()
        self.v_amp_in.setRange(0, 10000.0)
        self.v_amp_in.setValue(100.0)
        self.v_amp_in.setSingleStep(10.0)

        self.i_amp_in = QDoubleSpinBox()
        self.i_amp_in.setRange(0, 1000.0)
        self.i_amp_in.setValue(5.0)
        self.i_amp_in.setSingleStep(0.5)

        self.v3H_amp_in = QDoubleSpinBox()
        self.v3H_amp_in.setRange(0, 100)
        self.v3H_amp_in.setValue(0)
        self.v3H_amp_in.setSingleStep(1)

        self.v5H_amp_in = QDoubleSpinBox()
        self.v5H_amp_in.setRange(0, 100)
        self.v5H_amp_in.setValue(0)
        self.v5H_amp_in.setSingleStep(1)

        self.cos_phi_in = QDoubleSpinBox()
        self.cos_phi_in.setRange(0, 1)
        self.cos_phi_in.setValue(1)
        self.cos_phi_in.setSingleStep(0.05)

        form_layout.addRow("Network Interface:", self.iface_in)
        form_layout.addRow("Destination MAC:", self.dst_mac_in)
        form_layout.addRow("Source MAC:", self.src_mac_in)
        form_layout.addRow("VLAN ID (0=off):", self.vlan_id_in)
        form_layout.addRow("APPID (hex):", self.appid_in)
        form_layout.addRow("svID:", self.svid_in)
        form_layout.addRow("Frequency (Hz):", self.freq_in)
        form_layout.addRow("Samples/Cycle:", self.sps_in)
        form_layout.addRow("Voltage Amplitude (V):", self.v_amp_in)
        form_layout.addRow("Current Amplitude (A):", self.i_amp_in)
        form_layout.addRow("3th-harmonic Voltage (%):", self.v3H_amp_in)
        form_layout.addRow("5th-harmonic Voltage (%):", self.v5H_amp_in)
        form_layout.addRow("cos φ:", self.cos_phi_in)

        controls_layout.addLayout(form_layout)

        self.start_button = QPushButton("Start Simulation")
        self.start_button.clicked.connect(self.start_simulation)

        self.stop_button = QPushButton("Stop Simulation")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_simulation)

        self.save_button = QPushButton("Save Configuration")
        self.save_button.setEnabled(True)
        self.save_button.clicked.connect(self.save_config)

        self.load_button = QPushButton("Load Configuration")
        self.load_button.setEnabled(True)
        self.load_button.clicked.connect(self.load_config)

        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.save_button)
        controls_layout.addWidget(self.load_button)
        
        self.status_label = QLabel("Status: Stopped")
        self.packets_label = QLabel("Packets Sent: 0")
        controls_layout.addWidget(self.status_label)
        controls_layout.addWidget(self.packets_label)
        controls_layout.addStretch()
        controls_group.setLayout(controls_layout)

        # --- Plotting ---
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        pg.setConfigOption('background', 'w'); pg.setConfigOption('foreground', 'k')
        self.plot_v = pg.PlotWidget(title="Voltages")
        self.plot_i = pg.PlotWidget(title="Currents")
        plot_layout.addWidget(self.plot_v); plot_layout.addWidget(self.plot_i)
        self.plot_v.addLegend(); self.plot_i.addLegend()
        self.v_curves = {
            'Va': self.plot_v.plot(pen='r', name='Va'),
            'Vb': self.plot_v.plot(pen='g', name='Vb'),
            'Vc': self.plot_v.plot(pen='b', name='Vc')
        }
        self.i_curves = {
            'Ia': self.plot_i.plot(pen='r', name='Ia'),
            'Ib': self.plot_i.plot(pen='g', name='Ib'),
            'Ic': self.plot_i.plot(pen='b', name='Ic')
        }
        self.plot_data_v = np.zeros((500, 3)); self.plot_data_i = np.zeros((500, 3))
        main_layout.addWidget(plot_widget, 3)
        self.tabs.addTab(tab, "SV Publisher")

    def _create_decoder_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        decoder_group = QGroupBox("Decoder Controls")
        decoder_layout = QVBoxLayout()
        
        iface_layout = QHBoxLayout()
        iface_layout.addWidget(QLabel("Interface:"))
        self.decoder_iface_edit = QComboBox()
        self.decoder_iface_edit.addItems(get_if_list())
        iface_layout.addWidget(self.decoder_iface_edit)
        decoder_layout.addLayout(iface_layout)
        
        button_layout = QHBoxLayout()
        self.start_decode_button = QPushButton("Start Sniffing")
        self.stop_decode_button = QPushButton("Stop Sniffing")
        self.stop_decode_button.setEnabled(False)
        button_layout.addWidget(self.start_decode_button)
        button_layout.addWidget(self.stop_decode_button)
        decoder_layout.addLayout(button_layout)
        decoder_group.setLayout(decoder_layout)
        
        self.decoder_output = QTextEdit()
        self.decoder_output.setReadOnly(True)
        self.decoder_output.setStyleSheet("font-family: 'Courier New', monospace; color: #000000;")
        
        layout.addWidget(decoder_group)
        layout.addWidget(self.decoder_output)
        
        self.start_decode_button.clicked.connect(self.start_decoding)
        self.stop_decode_button.clicked.connect(self.stop_decoding)
        self.tabs.addTab(tab, "SV Decoder")

    def start_simulation(self):
        config = {
            "iface": self.iface_in.currentData(), "dst_mac": self.dst_mac_in.text(),
            "src_mac": self.src_mac_in.text(), "vlan_id": self.vlan_id_in.value(),
            "appid": self.appid_in.text(), "sv_id": self.svid_in.text(),
            "freq": int(self.freq_in.currentText()), "sps": int(self.sps_in.currentText()),
            "v_amp": self.v_amp_in.value(), "i_amp": self.i_amp_in.value(),
            "v3H_amp_in": self.v3H_amp_in.value(), "v5H_amp_in": self.v5H_amp_in.value(),
            "cos_phi": self.cos_phi_in.value()
        }
        self.thread = QThread(); self.worker = SVWorker(config)
        self.worker.moveToThread(self.thread)
        self.worker.new_data_signal.connect(self.update_plots)
        self.worker.packets_sent_signal.connect(self.update_packet_count)
        self.thread.started.connect(self.worker.run)
        self.thread.start()
        self.start_button.setEnabled(False); self.stop_button.setEnabled(True)
        self.status_label.setText("Status: Running")

    def stop_simulation(self):
        if self.worker: self.worker.stop()
        if self.thread: self.thread.quit(); self.thread.wait()
        self.start_button.setEnabled(True); self.stop_button.setEnabled(False)
        self.status_label.setText("Status: Stopped")

    def start_decoding(self):
        iface = self.decoder_iface_edit.currentText()
        self.decoder_output.clear()
        self.decoder_output.append(f"Starting sniffer on interface '{iface}'...")
        self.decoder_thread = QThread()
        self.decoder_worker = DecoderWorker(iface)
        self.decoder_worker.moveToThread(self.decoder_thread)
        self.decoder_worker.packet_decoded.connect(self.decoder_output.append)
        self.decoder_thread.started.connect(self.decoder_worker.run)
        self.decoder_thread.start()
        self.start_decode_button.setEnabled(False); self.stop_decode_button.setEnabled(True)
        self.decoder_iface_edit.setEnabled(False)

    def stop_decoding(self):
        if self.decoder_worker: self.decoder_worker.stop()
        if self.decoder_thread: self.decoder_thread.quit(); self.decoder_thread.wait()
        self.decoder_output.append("\nSniffer stopped.")
        self.start_decode_button.setEnabled(True); self.stop_decode_button.setEnabled(False)
        self.decoder_iface_edit.setEnabled(True)

    @pyqtSlot(np.ndarray)
    def update_plots(self, data_batch):
        # Correctly slice the data for plotting
        currents = data_batch[:, [0, 2, 4]] / 1000.0
        voltages = data_batch[:, [8, 10, 12]] / 100.0
        n = len(data_batch)
        
        self.plot_data_i = np.roll(self.plot_data_i, -n, axis=0)
        self.plot_data_i[-n:] = currents
        self.plot_data_v = np.roll(self.plot_data_v, -n, axis=0)
        self.plot_data_v[-n:] = voltages
        
        self.i_curves['Ia'].setData(self.plot_data_i[:, 0])
        self.i_curves['Ib'].setData(self.plot_data_i[:, 1])
        self.i_curves['Ic'].setData(self.plot_data_i[:, 2])
        self.v_curves['Va'].setData(self.plot_data_v[:, 0])
        self.v_curves['Vb'].setData(self.plot_data_v[:, 1])
        self.v_curves['Vc'].setData(self.plot_data_v[:, 2])

    @pyqtSlot(int)
    def update_packet_count(self, count):
        self.packets_label.setText(f"Packets Sent: {count}")

    def closeEvent(self, event):
        self.stop_simulation()
        self.stop_decoding()
        event.accept()

    def save_config(self):
        # Ścieżka do katalogu programu
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "config.xml")

        # Główny element XML
        root = ET.Element("SVSimulatorConfig")

        # Pomocnicza funkcja
        def add(parent, name, value):
            el = ET.SubElement(parent, name)
            el.text = str(value)

        # --- Parametry sieciowe ---
        network = ET.SubElement(root, "Network")
        add(network, "Interface", self.iface_in.currentData())
        add(network, "DestinationMAC", self.dst_mac_in.text())
        # add(network, "SourceMAC", self.src_mac_in.text())
        add(network, "VLAN_ID", self.vlan_id_in.value())
        add(network, "APPID", self.appid_in.text())
        add(network, "svID", self.svid_in.text())

        # --- Parametry próbkowania ---
        sampling = ET.SubElement(root, "Sampling")
        add(sampling, "Frequency_Hz", int(self.freq_in.currentText()))
        add(sampling, "SamplesPerCycle", int(self.sps_in.currentText()))

        # --- Parametry elektryczne ---
        electrical = ET.SubElement(root, "Electrical")
        add(electrical, "VoltageAmplitude_V", self.v_amp_in.value())
        add(electrical, "CurrentAmplitude_A", self.i_amp_in.value())
        add(electrical, "Voltage3H_percent", self.v3H_amp_in.value())
        add(electrical, "Voltage5H_percent", self.v5H_amp_in.value())
        add(electrical, "CosPhi", self.cos_phi_in.value())

        # Zapis do pliku
        tree = ET.ElementTree(root)
        tree.write(config_path, encoding="utf-8", xml_declaration=True)

        # Informacja w GUI (opcjonalnie)
        self.status_label.setText(f"Configuration saved to {config_path}")

    def load_config(self):
        # Katalog programu
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "config.xml")

        # Sprawdzenie czy plik istnieje
        if not os.path.exists(config_path):
            QMessageBox.warning(
                self,
                "Load Configuration",
                "Plik config.xml nie istnieje."
            )
            return

        try:
            tree = ET.parse(config_path)
            root = tree.getroot()

            # --- Network ---
            network = root.find("Network")
            if network is not None:

                saved_npf = network.findtext("Interface")

                iface_found = False

                if saved_npf:
                    for i in range(self.iface_in.count()):
                        if self.iface_in.itemData(i) == saved_npf:
                            self.iface_in.setCurrentIndex(i)
                            iface_found = True
                            break

                if not iface_found:
                    # fallback – zostaje aktualnie wybrany (np. pierwszy aktywny)
                    self.status_label.setText(
                        "Saved network interface not available – using current active interface"
                    )

                self.dst_mac_in.setText(network.findtext("DestinationMAC", ""))
                # self.src_mac_in.setText(network.findtext("SourceMAC", ""))
                self.vlan_id_in.setValue(int(network.findtext("VLAN_ID", 0)))
                self.appid_in.setText(network.findtext("APPID", ""))
                self.svid_in.setText(network.findtext("svID", ""))

            # --- Sampling ---
            sampling = root.find("Sampling")
            if sampling is not None:
                self.freq_in.setCurrentText(sampling.findtext("Frequency_Hz", "50"))
                self.sps_in.setCurrentText(sampling.findtext("SamplesPerCycle", "256"))

            # --- Electrical ---
            electrical = root.find("Electrical")
            if electrical is not None:
                self.v_amp_in.setValue(float(electrical.findtext("VoltageAmplitude_V", 0)))
                self.i_amp_in.setValue(float(electrical.findtext("CurrentAmplitude_A", 0)))
                self.v3H_amp_in.setValue(float(electrical.findtext("Voltage3H_percent", 0)))
                self.v5H_amp_in.setValue(float(electrical.findtext("Voltage5H_percent", 0)))
                self.cos_phi_in.setValue(float(electrical.findtext("CosPhi", 1)))

            self.status_label.setText("Configuration loaded from config.xml")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Load Configuration Error",
                f"Błąd podczas wczytywania konfiguracji:\n{e}"
            )

    # def _get_windows_iface_map(self):
    #     """
    #     Zwraca listę krotek:
    #     (display_name, npf_name)
    #     """
    #     npf_ifaces = get_if_list()
    #     win_ifaces = psutil.net_if_addrs()
    #
    #     result = []
    #
    #     for npf in npf_ifaces:
    #         if not npf.startswith(r"\Device\NPF_"):
    #             continue
    #
    #         guid = npf.replace(r"\Device\NPF_{", "").replace("}", "").upper()
    #         display = npf  # fallback
    #
    #         for win_name in win_ifaces.keys():
    #             if guid in win_name.upper():
    #                 display = win_name
    #                 break
    #
    #         result.append((display, npf))
    #
    #     return result

    @property
    def _get_active_windows_interfaces(self):
        """
        Zwraca listę krotek:
        (display_text, npf_name)
        TYLKO interfejsy:
        - status UP
        - mają adres IP
        """
        win_ifaces = scapy.arch.get_windows_if_list()
        stats = psutil.net_if_stats()
        addrs = psutil.net_if_addrs()

        result = []

        for iface in win_ifaces:
            name = iface.get("name")
            desc = iface.get("description", "")
            guid = iface.get("guid")

            if not isinstance(name,str) or not isinstance(guid,str):    #jeśli name lub guid nie jest stringiem
                continue                                                #pomiń i przejdź do kolejnego elementu pętli for

            # ✅ interfejs musi istnieć w psutil
            if name not in stats or name not in addrs:
                continue

            # ✅ musi być UP (fizycznie aktywny)
            if not stats[name].isup:
                continue

            # ✅ musi mieć adres IP (IPv4 lub IPv6)
            has_ip = any(
                addr.family.name in ("AF_INET", "AF_INET6")
                for addr in addrs[name]
            )
            if not has_ip:
                continue

            # ✅ poprawna nazwa dla Scapy
            npf = rf"\Device\NPF_{guid}"

            display = f"{name} – {desc}".strip()
            result.append((display, npf))

        return result



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
