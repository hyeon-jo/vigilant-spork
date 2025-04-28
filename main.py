# -*- coding: utf-8 -*-
from PySide6.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLineEdit, QLabel, QGridLayout, QWidget, QSizePolicy, QFrame,
                             QStatusBar, QProgressBar, QGroupBox, QSplitter, QTextEdit)
from PySide6.QtGui import QImage, QPixmap, QFont, QColor, QPalette
from PySide6.QtCore import Qt, Signal, Slot, QMutex, QTimer, QSize

import cv2
import os
import socket
import struct
import threading
import time
import numpy as np


IDX_SERVER_1 = 1
IDX_SERVER_2 = 2
IDX_SERVER_1_LOG = 3
IDX_SERVER_2_LOG = 4

LOGGING_STATUS_READY = 0
LOGGING_STATUS_SEND_MSG_TO_SERVER = 1
LOGGING_STATUS_RECEIVED_ACK_MSG_FROM_SERVER = 2

class Protocol_Header_org:
    def __init__(self, time_stamp=0, message_type=0, sequence_number=0, body_length=0):
        self.time_stamp = time_stamp
        self.message_type = message_type
        self.sequence_number = sequence_number
        self.body_length = body_length

    def serialize(self):
        fmt = "<QBQI"
        return struct.pack(fmt, self.time_stamp, self.message_type, self.sequence_number, self.body_length)

    @staticmethod
    def deserialize(data):
        fmt = "<QBQI"
        return struct.unpack(fmt, data)


# Protocol_Header 클래스 정의
class Protocol_Header:
    def __init__(self, time_stamp=0, message_type=0, sequence_number=0, body_length=0, mResult=0):
        self.time_stamp = time_stamp
        self.message_type = message_type
        self.sequence_number = sequence_number
        self.body_length = body_length
        self.mResult = mResult

    # Protocol_Header 객체를 바이너리 데이터로 직렬화합니다.
    def serialize(self):
        fmt = "<QBQIB"  # 리틀 엔디언, 8바이트 (uint64_t), 8바이트 (uint64_t), 1바이트 (uint8_t), 4바이트 (uint32_t), 1바이트 (uint8_t)
        return struct.pack(fmt, self.time_stamp, self.message_type, self.sequence_number, self.body_length, self.mResult)

    # 바이너리 데이터를 Protocol_Header 객체로 역직렬화합니다.
    @staticmethod
    def deserialize(data):
        fmt = "<QBQIB"
        return struct.unpack(fmt, data)

# Protocol_Header 클래스 정의
class Protocol_LoggingInfo:
    def __init__(self, time_stamp=0, message_type=0, sequence_number=0, body_length=0, time_stamp_log_start=0, time_stamp_log_end=0, metasize=0, metadescription=b'\x00' * 64):
        self.time_stamp = time_stamp
        self.message_type = message_type
        self.sequence_number = sequence_number
        self.body_length = body_length
        self.time_stamp_log_start = time_stamp_log_start
        self.time_stamp_log_end = time_stamp_log_end
        self.MetaSize = metasize
        # self.MetaDescription = metadescription
        self.MetaDescription = metadescription[:64]  
        
        
    # Protocol_Header 객체를 바이너리 데이터로 직렬화합니다.
    def serialize(self):
        fmt = "<QBQIQQI64s"  # 리틀 엔디언, 8바이트 (uint64_t), 8바이트 (uint64_t), 1바이트 (uint8_t), 4바이트 (uint32_t), 1바이트 (uint8_t)
        return struct.pack(fmt, self.time_stamp, self.message_type, self.sequence_number, self.body_length, self.time_stamp_log_start, self.time_stamp_log_end, self.MetaSize, self.MetaDescription)

    # 바이너리 데이터를 Protocol_Header 객체로 역직렬화합니다.
    @staticmethod
    def deserialize(data):
        fmt = "<QBQIQQI64s"
        return struct.unpack(fmt, data)


# Protocol_Header 클래스 정의
class Protocol_Inbound:
    def __init__(self, data_size):
        self.data_size = data_size        
        
    # Protocol_Header 객체를 바이너리 데이터로 직렬화합니다.
    def serialize(self):
        fmt = "<Q"  # 리틀 엔디언, 8바이트 (uint64_t), 8바이트 (uint64_t), 1바이트 (uint8_t), 4바이트 (uint32_t), 1바이트 (uint8_t)
        return struct.pack(fmt, self.data_size)

    # 바이너리 데이터를 Protocol_Header 객체로 역직렬화합니다.
    @staticmethod
    def deserialize(data):
        fmt = "<Q"
        return struct.unpack(fmt, data)



# Protocol_Header 클래스 정의
class LoggingStartMsg:
    def __init__(self, time_stamp=0, loggingMode=0):
        self.time_stamp = time_stamp
        self.loggingMode = loggingMode

    # Protocol_Header 객체를 바이너리 데이터로 직렬화합니다.
    def serialize(self):
        fmt = "<QB"  # 리틀 엔디언, 8바이트 (uint64_t), 8바이트 (uint64_t), 1바이트 (uint8_t), 4바이트 (uint32_t), 1바이트 (uint8_t)
        return struct.pack(fmt, self.time_stamp, self.loggingMode)

    # 바이너리 데이터를 Protocol_Header 객체로 역직렬화합니다.
    @staticmethod
    def deserialize(data):
        fmt = "<QB"
        return struct.unpack(fmt, data)


# Protocol_Header 클래스 정의
class DATA_SEND_REQ_Header:
    def __init__(self, time_stamp=0, message_type=17, sequence_number=0, body_length=8, request_status=0, data_type=0, sensor_channel= 0, service_id=0, network_id=0):
        self.time_stamp = time_stamp
        self.message_type = message_type
        self.sequence_number = sequence_number
        self.body_length = body_length

        self.request_status = request_status
        self.data_type = data_type
        self.sensor_channel = sensor_channel
        self.service_id = service_id
        self.network_id = network_id


    # Protocol_Header 객체를 바이너리 데이터로 직렬화합니다.
    def serialize(self):
        fmt = "<QBQIBBIBB"  # 리틀 엔디언, 8바이트 (uint64_t), 8바이트 (uint64_t), 1바이트 (uint8_t), 4바이트 (uint32_t), 1바이트 (uint8_t)
        return struct.pack(fmt, self.time_stamp, self.message_type, self.sequence_number, self.body_length, \
            self.request_status, self.data_type, self.sensor_channel, self.service_id, self.network_id)

    # 바이너리 데이터를 Protocol_Header 객체로 역직렬화합니다.
    @staticmethod
    def deserialize(data):
        fmt = "<QBQIBBIBB"
        return struct.unpack(fmt, data)


class ServerStatusIndicator(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(16, 16)
        self.set_status("disconnected")
        self.setToolTip("Server Connection Status")

    def set_status(self, status):
        if status == "connected":
            self.setStyleSheet("""
                QLabel {
                    background-color: qradialgradient(
                        cx: 0.5, cy: 0.5, radius: 0.8,
                        fx: 0.5, fy: 0.5,
                        stop: 0 #4cd964,
                        stop: 0.6 #2ecc71,
                        stop: 1 #27ae60
                    );
                    border-radius: 8px;
                    border: 1px solid #27ae60;
                }
                QLabel:hover {
                    background-color: qradialgradient(
                        cx: 0.5, cy: 0.5, radius: 0.8,
                        fx: 0.5, fy: 0.5,
                        stop: 0 #5cd964,
                        stop: 0.6 #3ecc71,
                        stop: 1 #37ae60
                    );
                }
            """)
        elif status == "disconnected":
            self.setStyleSheet("""
                QLabel {
                    background-color: qradialgradient(
                        cx: 0.5, cy: 0.5, radius: 0.8,
                        fx: 0.5, fy: 0.5,
                        stop: 0 #ff3b30,
                        stop: 0.6 #e74c3c,
                        stop: 1 #c0392b
                    );
                    border-radius: 8px;
                    border: 1px solid #c0392b;
                }
                QLabel:hover {
                    background-color: qradialgradient(
                        cx: 0.5, cy: 0.5, radius: 0.8,
                        fx: 0.5, fy: 0.5,
                        stop: 0 #ff4b40,
                        stop: 0.6 #f74c3c,
                        stop: 1 #d0392b
                    );
                }
            """)
        else:
            self.setStyleSheet("""
                QLabel {
                    background-color: qradialgradient(
                        cx: 0.5, cy: 0.5, radius: 0.8,
                        fx: 0.5, fy: 0.5,
                        stop: 0 #ffcc00,
                        stop: 0.6 #f1c40f,
                        stop: 1 #f39c12
                    );
                    border-radius: 8px;
                    border: 1px solid #f39c12;
                }
                QLabel:hover {
                    background-color: qradialgradient(
                        cx: 0.5, cy: 0.5, radius: 0.8,
                        fx: 0.5, fy: 0.5,
                        stop: 0 #ffdc00,
                        stop: 0.6 #f1d40f,
                        stop: 1 #f3ac12
                    );
                }
            """)

class VideoFrame(QLabel):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #2c3e50;
                border: 2px solid #34495e;
                border-radius: 5px;
                color: #ecf0f1;
                padding: 5px;
                margin: 2px;
            }
        """)
        self.setText(title)
        # self.setMinimumSize(180, 120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

class LiDARDisplay(QLabel):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #2c3e50;
                border: 2px solid #34495e;
                border-radius: 5px;
                color: #ecf0f1;
                padding: 5px;
                margin: 2px;
            }
        """)
        self.setText(title)
        self.setMinimumSize(160, 140)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

class LoggingInfoDisplay(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setStyleSheet("""
            QTextEdit {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 2px solid #34495e;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        self.setMinimumHeight(100)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

class ClientDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HADF Logging Application")
        self.resize(1280, 800)
        self.setMinimumSize(1024, 768)
        
        # 타이머 설정
        self.timer = QTimer()
        self.timer.timeout.connect(self.send_divide_signal)
        self.timestamp = int(time.time_ns()) + 30000000000

        # 메인 레이아웃
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 상단 컨트롤 패널과 로깅 정보를 수평으로 배치
        top_layout = QHBoxLayout()
        
        # 상단 컨트롤 패널
        control_panel = QGroupBox("Control Panel")
        control_layout = QHBoxLayout()
        
        # 서버 연결 설정
        server_group = QGroupBox("Server Settings")
        server_layout = QGridLayout()
        
        server1_label = QLabel("Server 1 IP:")
        self.server1_ip_edit = QLineEdit("192.168.10.102")
        self.server1_status_indicator = ServerStatusIndicator()
        self.server1_status_indicator_log = ServerStatusIndicator()
        
        server2_label = QLabel("Server 2 IP:")
        self.server2_ip_edit = QLineEdit("192.168.10.202")
        self.server2_status_indicator = ServerStatusIndicator()
        self.server2_status_indicator_log = ServerStatusIndicator()

        # 메타 설명 입력
        meta_label = QLabel("Meta Description:")
        self.meta_description_edit = QLineEdit()
        self.meta_description_edit.setPlaceholderText("Enter meta description (max 64 characters)")
        self.meta_description_edit.setMaxLength(64)
        
        # 메타 설명 전송 버튼
        self.send_meta_button = QPushButton("Send Meta")
        self.send_meta_button.clicked.connect(self.send_meta_description)
        
        server_layout.addWidget(server1_label, 0, 0)
        server_layout.addWidget(self.server1_ip_edit, 0, 1)
        server_layout.addWidget(self.server1_status_indicator, 0, 2)
        server_layout.addWidget(self.server1_status_indicator_log, 0, 3)
        server_layout.addWidget(server2_label, 1, 0)
        server_layout.addWidget(self.server2_ip_edit, 1, 1)
        server_layout.addWidget(self.server2_status_indicator, 1, 2)
        server_layout.addWidget(self.server2_status_indicator_log, 1, 3)
        server_layout.addWidget(meta_label, 2, 0)
        server_layout.addWidget(self.meta_description_edit, 2, 1, 1, 2)
        server_layout.addWidget(self.send_meta_button, 2, 3)
        server_group.setLayout(server_layout)
        
        # 버튼 그룹
        button_group = QGroupBox("Control")
        button_layout = QVBoxLayout()
        button_layout.setSpacing(5)  # 버튼 사이의 간격 설정
        
        self.connect_button = QPushButton("Connect")
        self.logging_start_button = QPushButton("Start Logging")
        self.exit_button = QPushButton("Exit")
        
        self.connect_button.clicked.connect(self.on_connect_clicked)
        self.logging_start_button.clicked.connect(self.on_logging_start_clicked)
        self.exit_button.clicked.connect(self.close)
        
        button_layout.addWidget(self.connect_button)
        button_layout.addWidget(self.logging_start_button)
        button_layout.addWidget(self.exit_button)
        button_group.setLayout(button_layout)
        
        control_layout.addWidget(server_group)
        control_layout.addWidget(button_group)
        control_panel.setLayout(control_layout)
        
        # 로깅 정보 표시
        logging_info_group = QGroupBox("Logging Information")
        logging_info_layout = QVBoxLayout()
        self.logging_info_display = LoggingInfoDisplay()
        logging_info_layout.addWidget(self.logging_info_display)
        logging_info_group.setLayout(logging_info_layout)
        
        # Control Panel과 Logging Information을 수평으로 배치
        top_layout.addWidget(control_panel)
        top_layout.addWidget(logging_info_group)
        
        # 비디오 프레임 그리드
        video_group = QGroupBox("Video Stream")
        video_layout = QGridLayout()
        video_layout.setContentsMargins(5, 20, 5, 5)
        video_layout.setSpacing(5)
        self.video_labels = []
        for i in range(12):
            label = VideoFrame(f"Camera {i+1}")
            self.video_labels.append(label)
            video_layout.addWidget(label, i // 4, i % 4)
        video_group.setLayout(video_layout)
        
        # LiDAR 디스플레이
        lidar_group = QGroupBox("LiDAR Data")
        lidar_layout = QHBoxLayout()
        lidar_layout.setContentsMargins(5, 20, 5, 5)
        lidar_layout.setSpacing(5)
        self.lidar_displays = []
        for i in range(5):
            lidar_display = LiDARDisplay(f"LiDAR {i+1}")
            self.lidar_displays.append(lidar_display)
            lidar_layout.addWidget(lidar_display)
        lidar_group.setLayout(lidar_layout)
        
        # 비디오와 LiDAR를 수평으로 배치
        visualization_layout = QHBoxLayout()
        visualization_layout.addWidget(video_group)
        visualization_layout.addWidget(lidar_group)
        
        # 레이아웃 조립
        main_layout.addLayout(top_layout)
        main_layout.addLayout(visualization_layout)
        
        self.setLayout(main_layout)
        
        # 상태 추적
        self.is_logging = False
        self.server1_socket = None
        self.server2_socket = None
        self.server1_socket_log = None
        self.server2_socket_log = None
        self.ip1_status = 0
        self.ip2_status = 0
        self.mutex_send = QMutex()
        self.mutex_receive = QMutex()

        self.is_camera_on_server_1 = False
        self.is_camera_on_server_2 = False

        self.mutex_connect_server = QMutex()

        self.server1_socket_send_thread = None
        self.server1_socket_receive_thread = None
        self.server2_socket_send_thread = None
        self.server2_socket_receive_thread = None
        

        self.server_socket = []
        self.server_socket_log = []
        self.server_socket_receive_thread = []
        self.server_socket_send_thread = []
        self.server_socket_receive_thread_log = []

        self.server_socket.append(None)
        self.server_socket.append(None)

        self.server_socket_log.append(None)
        self.server_socket_log.append(None)

        self.server_1_logging_status = LOGGING_STATUS_READY
        self.server_2_logging_status = LOGGING_STATUS_READY
        

        # 테마 적용
        self.apply_modern_theme()

    # START

    @Slot()
    def on_connect_clicked(self):
        ip1 = self.server1_ip_edit.text()
        ip2 = self.server2_ip_edit.text()

        port = 9090
        log_port = 9091

        # 서버 연결 시도
        threading.Thread(target=self.connect_to_server1, args=(ip1, port, self.server1_status_indicator, IDX_SERVER_1)).start()
        threading.Thread(target=self.connect_to_server2, args=(ip2, port, self.server2_status_indicator, IDX_SERVER_2)).start()
        threading.Thread(target=self.connect_to_server_log1, args=(ip1, log_port, self.server1_status_indicator_log, IDX_SERVER_1_LOG)).start()
        threading.Thread(target=self.connect_to_server_log2, args=(ip2, log_port, self.server2_status_indicator_log, IDX_SERVER_2_LOG)).start()

    def connect_to_server1(self, ip, port, status_indicator, server_id):
        try:
            self.mutex_connect_server.lock()

            sock_server_1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock_server_1.connect((ip, port))
            status_indicator.set_status("connected")
            print(f"Successfully connected to Server: {ip} {port}")

            self.server_socket[0] = sock_server_1
            self.server1_socket_receive_thread = threading.Thread(target=self.receive_data_server_1, args=(sock_server_1, server_id))
            self.server1_socket_receive_thread.start()
            # self.server1_socket_send_thread = threading.Thread(target=self.send_data, args=(sock_server_1, server_id))
            # self.server1_socket_send_thread.start()
        
            self.send_LINK_REQ(sock_server_1)

            self.mutex_connect_server.unlock()
        except Exception as e:
            self.mutex_connect_server.unlock()
            status_indicator.set_status("disconnected")
            print(f"**************** Failed to connect to Server: {ip}. Error: {e}")

    def connect_to_server2(self, ip, port, status_indicator, server_id):
        try:
            self.mutex_connect_server.lock()

            sock_server_2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock_server_2.connect((ip, port))
            status_indicator.set_status("connected")
            print(f"Successfully connected to Server: {ip} {port}")

            self.server_socket[1] = sock_server_2
            self.server2_socket_receive_thread = threading.Thread(target=self.receive_data_server_2, args=(sock_server_2, server_id))
            self.server2_socket_receive_thread.start()
            # self.server2_socket_send_thread = threading.Thread(target=self.send_data, args=(sock_server_2, server_id))
            # self.server2_socket_send_thread.start()

            self.send_LINK_REQ(sock_server_2)

            self.mutex_connect_server.unlock()
        except Exception as e:
            self.mutex_connect_server.unlock()
            status_indicator.set_status("disconnected")
            print(f"**************** Failed to connect to Server: {ip}. Error: {e}")

    def connect_to_server_log1(self, ip, port, status_indicator, server_id):
        try:
            self.mutex_connect_server.lock()

            sock_server_log_1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock_server_log_1.connect((ip, port))
            status_indicator.set_status("connected")
            print(f"Successfully connected to Server: {ip} {port}")

            self.server_socket_log[0] = sock_server_log_1

            self.mutex_connect_server.unlock()
        except Exception as e:
            self.mutex_connect_server.unlock()
            status_indicator.set_status("disconnected")
            print(f"**************** Failed to connect to Server: {ip}. Error: {e}")

    def connect_to_server_log2(self, ip, port, status_indicator, server_id):
        try:
            self.mutex_connect_server.lock()

            sock_server_log_2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock_server_log_2.connect((ip, port))
            status_indicator.set_status("connected")
            print(f"Successfully connected to Server: {ip} {port}")

            self.server_socket_log[1] = sock_server_log_2
            
            self.mutex_connect_server.unlock()
        except Exception as e:
            self.mutex_connect_server.unlock()
            status_indicator.set_status("disconnected")
            print(f"**************** Failed to connect to Server: {ip}. Error: {e}")

    def on_disconnect_clicked(self):
        try:
            if self.server_socket[0]:
                self.server_socket[0].close()  # Close server1 socket if it exists
                self.server_socket[0] = None

            if self.server_socket[1]:
                self.server_socket[1].close()  # Close server2 socket if it exists
                self.server_socket[1] = None

            
            if self.server_socket_log[0]:
                self.server_socket_log[0].close()  # Close server1 socket if it exists
                self.server_socket_log[0] = None

            if self.server_socket_log[1]:
                self.server_socket_log[1].close()  # Close server2 socket if it exists
                self.server_socket_log[1] = None
                
            if self.server1_socket_send_thread != None:
                self.server1_socket_send_thread.join()
            if self.server2_socket_send_thread != None:
                self.server2_socket_send_thread.join()

            if self.server1_socket_receive_thread != None:
                self.server1_socket_receive_thread.join()
            if self.server2_socket_receive_thread != None:
                self.server2_socket_receive_thread.join()


        except Exception as e:
            print(f"**************** Failed to disconnect server Error: {e}")

    def send_LINK_REQ(self, sock):
        current_time = int(time.time_ns())  # Current timestamp in milliseconds

        header = Protocol_Header(
            time_stamp=current_time, 
            message_type=1,     # LINK_REQ
            sequence_number=0, 
            body_length=1,
            mResult=0)

        # 헤더 직렬화
        serialized_data = header.serialize()

        # 직렬화된 데이터 보내기
        sock.sendall(serialized_data)
        
        print(f'\n {sock} : [send_LINK_REQ] Header sent to client:', current_time)

    def send_REC_INFO_REQ(self, sock):
        current_time = int(time.time_ns())  # Current timestamp in milliseconds
        header = Protocol_Header(
            time_stamp=current_time, 
            message_type=3,     # REC_INFO_REQ
            sequence_number=0, 
            body_length=1,
            mResult=0)

        # 헤더 직렬화
        serialized_data = header.serialize()

        # 직렬화된 데이터 보내기
        sock.sendall(serialized_data)
        
        print('[send_REC_INFO_REQ] Header sent to client:', current_time)

    def send_SEND_REQ(self, sock, sensor_on):

        current_time = int(time.time_ns())

            # send DATA_SEND_REQ
        if sensor_on == 0:
            header = DATA_SEND_REQ_Header(
                time_stamp=current_time, 
                message_type=17,
                sequence_number=0, 
                body_length=8,
                request_status = 1,
                data_type = 1,
                sensor_channel = 4294967295,
                service_id = 0, 
                network_id = 0)
        else:
            header = DATA_SEND_REQ_Header(
                time_stamp=current_time, 
                message_type=17,
                sequence_number=0, 
                body_length=8,
                request_status = 0,
                data_type = 1,
                sensor_channel = 4294967295,
                service_id = 0, 
                network_id = 0)

        print(f'[send_SEND_REQ] {sock} sent to client:', current_time, "sensor channel:", header.sensor_channel)
        # 헤더 직렬화
        serialized_data = header.serialize()
        # 직렬화된 데이터 보내기
        sock.sendall(serialized_data)
        # print('DATA_SEND_REQ_Header sent to client:', header.deserialize())

    def receive_data_server_1(self, sock, server_id):
        try:
            while sock:
                # self.mutex_receive.lock()

                # Receive the header from the server
                response_data = sock.recv(21)
                print(f"[{sock}]", end=': ')
                received_size = len(response_data)
                response_header = []
                if received_size == 21:
                    # 응답 데이터 역직렬화
                    response_header = Protocol_Header_org.deserialize(response_data)

                    received_data_header =  {
                        "TimeStamp": response_header[0],
                        "MessageType": response_header[1],
                        "SequenceNumber": response_header[2],
                        "BodyLength": response_header[3]
                    }

                    body_length = int(received_data_header['BodyLength'])

                    recv_data = b""
                    data_tmp = b""
                    while True:
                        if (body_length) - len(recv_data) < 1024:
                            data_tmp = sock.recv((body_length) - len(recv_data))
                            # print("remained data size: ", len(data_tmp), (dese[3] - 1) - len(recv_data))
                            recv_data += data_tmp
                            break

                        data_tmp = sock.recv(1024)
                        # print("!! remained data size: ", len(data_tmp), (dese[3] - 1) - len(recv_data))
                        recv_data += data_tmp

                    print(f"[receive_data] Header: TimeStamp={response_header[0]}, MessageType={response_header[1]}, SequenceNumber={response_header[2]}, BodyLength={response_header[3]}, received total data size {len(recv_data)}")

                    if received_data_header['MessageType'] == 2:
                        self.send_REC_INFO_REQ(sock)
                        self.send_SEND_REQ(sock, 0)
                    elif received_data_header['MessageType'] == 5:

                        # # current_time = int(time.time_ns()) 
                        # # if current_time - self.pre_timestamp_for_drawing < 1000000000:
                        # #     print(current_time, self.pre_timestamp_for_drawing, current_time - self.pre_timestamp_for_drawing)
                        # #     continue

                        # self.pre_timestamp_for_drawing = current_time

                        # 구조체 정의
                        SEND_DATA_HEADER_FORMAT = "IIIIQBBHHBBII"
                        SEND_DATA_HEADER_SIZE = struct.calcsize(SEND_DATA_HEADER_FORMAT)

                        # 헤더 파싱
                        header = struct.unpack(SEND_DATA_HEADER_FORMAT, recv_data[:SEND_DATA_HEADER_SIZE])
                        
                        send_data_header =  {
                            "mSequenceNumber": header[0],
                            "mTotalNumber": header[1],
                            "mCurrentNumber": header[2],
                            "mFrameNumber": header[3],
                            "mTimestamp": header[4],
                            "mSensorType": header[5],
                            "mChannel": header[6],
                            "mImgWidth": header[7],
                            "mImgHeight": header[8],
                            "mImgDepth": header[9],
                            "mImgFormat": header[10],
                            "mNumPoints": header[11],
                            "mPayloadSize": header[12]
                        }
                        # print(send_data_header)
                        
                        # pre_data = client_socket.recv(30)
                        # data = client_socket.recv(1024)
                        if send_data_header['mSensorType'] == 1:    # Sensor: CAMERA
                                
                            image_buffer = recv_data[SEND_DATA_HEADER_SIZE:]

                            # 광각 영상에 대해서만 처리
                            # height = 614
                            # width = 768
                            
                            grayscale_image = self.yuv422uyvy_to_grayscale(image_buffer, send_data_header['mImgWidth'], send_data_header['mImgHeight'])

                            # cv2.resize(grayscale_image, (int(grayscale_image.shape[1] / 4), int(grayscale_image.shape[0] / 4)))

                            # cv2.imwrite(str("src/python/test_grayscale_" + str(header[4]) + ".jpg"), grayscale_image)
                            # cv2.imshow(str("test" + str(header[4])), grayscale_image)
                            # cv2.waitKey(10)
                            print(str("received image data: ch " + str(header[4])))
                            self.update_video_frame(grayscale_image, send_data_header['mChannel'])


                        if send_data_header['mSensorType'] == 2:    # Sensor: LIDAR
                            point_cloud_buffer = recv_data[SEND_DATA_HEADER_SIZE:]
                            print("point_cloud_buffer length", point_cloud_buffer.__len__())
                            print("mNumPoints", send_data_header['mNumPoints'])

                            point_cloud_data = np.frombuffer(point_cloud_buffer, dtype=np.float32).reshape((send_data_header['mNumPoints'], 4))
                            # print(point_cloud_data[0:10])
                    elif received_data_header['MessageType'] == 25:
                        print("\n *** ----------------- logging success", "server id:", server_id, "\n")
                        if server_id == IDX_SERVER_1:
                            self.server_1_logging_status = LOGGING_STATUS_RECEIVED_ACK_MSG_FROM_SERVER
                        if server_id == IDX_SERVER_2:
                            self.server_2_logging_status = LOGGING_STATUS_RECEIVED_ACK_MSG_FROM_SERVER

                    # self.mutex_receive.unlock()

                else:
                    print("************ [receive_data] Error !!!! size:", len(response_data), "response_data", response_data)
                    
                
        except Exception as e:
            print(f"**************** Failed to receive data from Server {server_id}. Error: {e}")
            # self.mutex_receive.unlock()
        finally:
            sock.close()
            print(f"Connection closed with Server {server_id}")
            if server_id == 1:
                self.server1_status_indicator.set_status("disconnected")
                self.server_socket[0] = None
            elif server_id == 2:
                self.server2_status_indicator.set_status("disconnected")
                self.server_socket[1] = None
            
            # elif server_id == 3:
            #     self.server1_status_indicator_log.set_status("disconnected")
            #     self.server_socket_log[0] = None
            # elif server_id == 4:
            #     self.server2_status_indicator_log.set_status("disconnected")
            #     self.server_socket_log[1] = None

    def receive_data_server_2(self, sock, server_id):
        try:
            while sock:
                # self.mutex_receive.lock()

                # Receive the header from the server
                response_data = sock.recv(21)
                print(f"[{sock}]", end=': ')
                received_size = len(response_data)
                response_header = []
                if received_size == 21:
                    # 응답 데이터 역직렬화
                    response_header = Protocol_Header_org.deserialize(response_data)

                    received_data_header =  {
                        "TimeStamp": response_header[0],
                        "MessageType": response_header[1],
                        "SequenceNumber": response_header[2],
                        "BodyLength": response_header[3]
                    }

                    body_length = int(received_data_header['BodyLength'])

                    recv_data = b""
                    data_tmp = b""
                    while True:
                        if (body_length) - len(recv_data) < 1024:
                            data_tmp = sock.recv((body_length) - len(recv_data))
                            # print("remained data size: ", len(data_tmp), (dese[3] - 1) - len(recv_data))
                            recv_data += data_tmp
                            break

                        data_tmp = sock.recv(1024)
                        # print("!! remained data size: ", len(data_tmp), (dese[3] - 1) - len(recv_data))
                        recv_data += data_tmp

                    print(f"[receive_data] Header: TimeStamp={response_header[0]}, MessageType={response_header[1]}, SequenceNumber={response_header[2]}, BodyLength={response_header[3]}, received total data size {len(recv_data)}")

                    if received_data_header['MessageType'] == 2:
                        self.send_REC_INFO_REQ(sock)
                        self.send_SEND_REQ(sock, 0)
                    elif received_data_header['MessageType'] == 5:
                        # 구조체 정의
                        SEND_DATA_HEADER_FORMAT = "IIIIQBBHHBBII"
                        SEND_DATA_HEADER_SIZE = struct.calcsize(SEND_DATA_HEADER_FORMAT)

                        # 헤더 파싱
                        header = struct.unpack(SEND_DATA_HEADER_FORMAT, recv_data[:SEND_DATA_HEADER_SIZE])
                        
                        send_data_header =  {
                            "mSequenceNumber": header[0],
                            "mTotalNumber": header[1],
                            "mCurrentNumber": header[2],
                            "mFrameNumber": header[3],
                            "mTimestamp": header[4],
                            "mSensorType": header[5],
                            "mChannel": header[6],
                            "mImgWidth": header[7],
                            "mImgHeight": header[8],
                            "mImgDepth": header[9],
                            "mImgFormat": header[10],
                            "mNumPoints": header[11],
                            "mPayloadSize": header[12]
                        }
                        # print(send_data_header)
                        
                        # pre_data = client_socket.recv(30)
                        # data = client_socket.recv(1024)
                        if send_data_header['mSensorType'] == 1:    # Sensor: CAMERA
                                
                            image_buffer = recv_data[SEND_DATA_HEADER_SIZE:]

                            # 광각 영상에 대해서만 처리
                            # height = 614
                            # width = 768
                            
                            grayscale_image = self.yuv422uyvy_to_grayscale(image_buffer, send_data_header['mImgWidth'], send_data_header['mImgHeight'])
                            # cv2.imwrite(str("src/python/test_grayscale_" + str(header[4]) + ".jpg"), grayscale_image)
                            # cv2.imshow(str("test" + str(header[4])), grayscale_image)
                            # cv2.waitKey(10)
                            print(str("received image data: ch " + str(header[4])))
                            self.update_video_frame(grayscale_image, send_data_header['mChannel'])


                        if send_data_header['mSensorType'] == 2:    # Sensor: LIDAR
                            point_cloud_buffer = recv_data[SEND_DATA_HEADER_SIZE:]
                            print("point_cloud_buffer length", point_cloud_buffer.__len__())
                            print("mNumPoints", send_data_header['mNumPoints'])

                            point_cloud_data = np.frombuffer(point_cloud_buffer, dtype=np.float32).reshape((send_data_header['mNumPoints'], 4))
                            # print(point_cloud_data[0:10])
                    elif received_data_header['MessageType'] == 25:
                        print("\n *** ----------------- logging success", "server id:", server_id, "\n")
                        if server_id == IDX_SERVER_1:
                            self.server_1_logging_status = LOGGING_STATUS_RECEIVED_ACK_MSG_FROM_SERVER
                        if server_id == IDX_SERVER_2:
                            self.server_2_logging_status = LOGGING_STATUS_RECEIVED_ACK_MSG_FROM_SERVER

                    # self.mutex_receive.unlock()

                else:
                    print("************ [receive_data] Error !!!! size:", len(response_data), "response_data", response_data)
                    
                
        except Exception as e:
            print(f"**************** Failed to receive data from Server {server_id}. Error: {e}")
            # self.mutex_receive.unlock()
        finally:
            sock.close()
            print(f"Connection closed with Server {server_id}")
            if server_id == 1:
                self.server1_status_indicator.set_status("disconnected")
                self.server_socket[0] = None
            elif server_id == 2:
                self.server2_status_indicator.set_status("disconnected")
                self.server_socket[1] = None
            
            # elif server_id == 3:
            #     self.server1_status_indicator_log.set_status("disconnected")
            #     self.server_socket_log[0] = None
            # elif server_id == 4:
            #     self.server2_status_indicator_log.set_status("disconnected")
            #     self.server_socket_log[1] = None

    def receive_data_log(self, sock, server_id):
        try:
            while sock:
                
                # self.mutex_receive_log.lock()

                response_data = sock.recv(21)

                received_size = len(response_data)
                print("!!!!!! [receive_data_log] received size: ", received_size)

                if received_size == 0:
                    continue

                fmt = "<QBQI"
                # dese = struct.unpack(fmt, response_data)
                dese = struct.unpack(fmt, response_data[0:21])
                print(dese)

                recv_data = b""
                data_tmp = b""
                while True:
                    if (dese[3]) - len(recv_data) < 1024:
                        data_tmp = sock.recv((dese[3]) - len(recv_data))
                        recv_data += data_tmp
                        break

                    data_tmp = sock.recv(1024)
                    recv_data += data_tmp
                    
                print(" recv_data total received size: ", len(recv_data))

                # self.mutex_receive_log.unlock()

                        
        except Exception as e:
            print(f"**************** Failed to receive_data_log data from Server {server_id}. Error: {e}")
            if server_id == 3:
                self.server1_status_indicator_log.set_status("disconnected")
                self.server_socket_log[0] = None
            elif server_id == 4:
                self.server2_status_indicator_log.set_status("disconnected")
                self.server_socket_log[1] = None
        finally:
            sock.close()
            print(f"Connection closed with Server {server_id}")
            if server_id == 3:
                self.server1_status_indicator_log.set_status("disconnected")
                self.server_socket_log[0] = None
            elif server_id == 4:
                self.server2_status_indicator_log.set_status("disconnected")
                self.server_socket_log[1] = None
            
            # elif server_id == 3:
            #     self.server1_status_indicator_log.set_status("disconnected")
            #     self.server_socket_log[0] = None
            # elif server_id == 4:
            #     self.server2_status_indicator_log.set_status("disconnected")
            #     self.server_socket_log[1] = None

    def send_StartLoggingMessageToServer(self):
        try:
            current_time = int(time.time_ns())  # Current timestamp in milliseconds

            # HADF_LOGGING_START_CONT_REQ 19

            header = Protocol_LoggingInfo(
                time_stamp=current_time, 
                message_type=19,     # Logging start
                sequence_number=0, 
                body_length=84,
                time_stamp_log_start=current_time,
                time_stamp_log_end=0,
                metasize=64,
                metadescription=str("aaaaaa").encode('utf-8')
                )

            print("[send_StartLoggingMessageToServer] current_time:", current_time, "messageType", 19)

            # 헤더 직렬화
            serialized_data = header.serialize()

            # 직렬화된 데이터 보내기
            if self.server_socket_log[0]:
                self.server_1_logging_status = LOGGING_STATUS_SEND_MSG_TO_SERVER
                self.server_socket_log[0].sendall(serialized_data)

            if self.server_socket_log[1]:
                
                self.server_2_logging_status = LOGGING_STATUS_SEND_MSG_TO_SERVER
                self.server_socket_log[1].sendall(serialized_data)
            
            curTime = int(time.time())
            if self.server_socket_log[0]:
                while True:
                    time.sleep(0.5)
                    print("\n**************** wait for receive start logging message: 1\n")
                    # if int(time.time()) - curTime > 3:
                    #     print("\n\n**************** Failed to logging on Server: 1\n\n")
                    #     break

                    if self.server_1_logging_status == LOGGING_STATUS_RECEIVED_ACK_MSG_FROM_SERVER:
                        self.server_1_logging_status = LOGGING_STATUS_READY
                        print("*************!!! received start logging message: 1\n")
                        break

            if self.server_socket_log[1]:
                while True:
                    time.sleep(0.5)
                    print("\n**************** wait for receive start logging message: 2\n")
                    # if int(time.time()) - curTime > 3:
                    #     print("\n\n**************** Failed to logging on Server: 2\n\n")
                    #     break
                    
                    if self.server_2_logging_status == LOGGING_STATUS_RECEIVED_ACK_MSG_FROM_SERVER:
                        self.server_2_logging_status = LOGGING_STATUS_READY
                        print("*************!!! received start logging message: 2\n")
                        break
            
        except Exception as e:
            print(f"**************** Failed to connect to Server: . Error: {e}")

    def send_EndLoggingMessageToServer(self):
        try:
            current_time = int(time.time_ns())  # Current timestamp in milliseconds

            # HADF_LOGGING_START_CONT_REQ 19

            header = Protocol_LoggingInfo(
                time_stamp=current_time, 
                message_type=20,     # Logging start
                sequence_number=0, 
                body_length=84,
                time_stamp_log_start=0,
                time_stamp_log_end=current_time,
                metasize=64,
                metadescription=str("aaaaaa").encode('utf-8')
                )

            print("[send_EndLoggingMessageToServer] current_time:", current_time, "messageType", 20)

            # 헤더 직렬화
            serialized_data = header.serialize()

            # 직렬화된 데이터 보내기
            if self.server_socket_log[0]:
                self.server_1_logging_status = LOGGING_STATUS_SEND_MSG_TO_SERVER
                self.server_socket_log[0].sendall(serialized_data)

            time.sleep(1)

            if self.server_socket_log[1]:
                self.server_2_logging_status = LOGGING_STATUS_SEND_MSG_TO_SERVER
                self.server_socket_log[1].sendall(serialized_data)

            curTime = int(time.time())
            if self.server_socket_log[0]:
                while True:
                    time.sleep(0.5)
                    print("\n**************** wait for receive end logging message : 1\n")

                    if self.server_1_logging_status == LOGGING_STATUS_RECEIVED_ACK_MSG_FROM_SERVER:
                        self.server_1_logging_status = LOGGING_STATUS_READY
                        print("*************!!! received end logging message: 1\n")
                        break

            if self.server_socket_log[1]:
                while True:
                    time.sleep(0.5)
                    print("\n**************** wait for receive end logging message: 2\n")
                    
                    if self.server_2_logging_status == LOGGING_STATUS_RECEIVED_ACK_MSG_FROM_SERVER:
                        self.server_2_logging_status = LOGGING_STATUS_READY
                        print("*************!!! received end logging message: 2\n")
                        break
            
            
        except Exception as e:
            print(f"**************** Failed to connect to Server: . Error: {e}")
    # END

    def apply_modern_theme(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a1a;
                color: #ecf0f1;
            }
            QGroupBox {
                border: 2px solid #34495e;
                border-radius: 5px;
                margin-top: 1em;
                padding-top: 10px;
                color: #ecf0f1;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #2472a4;
            }
            QLineEdit {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 2px solid #34495e;
                border-radius: 4px;
                padding: 5px;
            }
            QLabel {
                color: #ecf0f1;
            }
        """)

    def send_divide_signal(self):
        print("HERE!!")
        if self.is_logging:
            self.timestamp = self.timestamp + 30000000000 # 30 seconds
            # self.send_logging_signal(message_type=20)
            self.send_EndLoggingMessageToServer()
            # self.send_logging_signal(message_type=19)
            self.send_StartLoggingMessageToServer()

    def send_meta_description(self):
        try:
            # Get meta description from UI
            meta_description = self.meta_description_edit.text().encode('utf-8')
            if len(meta_description) > 64:
                meta_description = meta_description[:64]
            elif len(meta_description) < 64:
                meta_description = meta_description + b'\x00' * (64 - len(meta_description))

            header = Protocol_LoggingInfo(
                time_stamp=self.timestamp, 
                message_type=19,     # Logging start
                sequence_number=0, 
                body_length=84,
                time_stamp_log_start=self.timestamp,
                time_stamp_log_end=0,
                metasize=64,
                metadescription=meta_description
                )

            # 헤더 직렬화
            serialized_data = header.serialize()

            # 직렬화된 데이터 보내기
            if self.server1_socket_log:
                self.server1_socket_log.sendall(serialized_data)

            if self.server2_socket_log:
                self.server2_socket_log.sendall(serialized_data)
            
            # 입력창 초기화
            self.meta_description_edit.clear()
            
            # 로깅 정보 업데이트
            meta_text = meta_description.decode('utf-8').replace('\x00', '')
            self.logging_info_display.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Meta description sent: {meta_text}")
            
            print("Meta description sent successfully")
            
        except Exception as e:
            print(f"Failed to send meta description: {e}")
            self.logging_info_display.append(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error: {str(e)}")

    @Slot()
    def on_logging_start_clicked(self):
        if not self.is_logging:
            self.logging_start_button.setText("Stop Logging")
            self.is_logging = True
            self.timestamp = int(time.time_ns())
            
            # self.send_logging_signal(message_type=19)
            self.send_StartLoggingMessageToServer()
            self.timer.start(29900)
        else:
            self.logging_start_button.setText("Start Logging")
            self.timer.stop()
            # self.send_logging_signal(message_type=20)
            self.send_EndLoggingMessageToServer()
            self.is_logging = False

    @Slot()
    def update_video_frame(self, frame, idx):
        # # Update video frames here
        # for i, frame in enumerate(frames):
        #     if idx is not i:
        #         continue

        #     if i < len(self.video_labels):
        #         frame = np.ascontiguousarray(frame)
        #         image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_Grayscale8)
        #         pixmap = QPixmap.fromImage(image)
        #         self.video_labels[i].setPixmap(pixmap.scaled(self.video_labels[i].size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        frame = np.ascontiguousarray(frame)
        image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(image)
        self.video_labels[idx].setPixmap(pixmap.scaled(self.video_labels[idx].size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def yuv422uyvy_to_grayscale(self, raw_data, width, height):
        # YUV422UYVY 데이터를 NumPy 배열로 변환합니다.
        yuv_image = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width * 2))
        
        # Y값만 추출합니다. UYVY 포맷에서 Y값은 홀수 인덱스에 위치합니다.
        Y_values_odd = yuv_image[1::2, 1::4]
        # Y_values_even = yuv_image[:, 3::4]
        
        # 두 Y값의 평균을 계산하여 그레이스케일 이미지로 만듭니다.
        # grayscale_image = (Y_values_odd + Y_values_even) // 2
        grayscale_image = Y_values_odd
        
        return grayscale_image

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        super().keyPressEvent(event)

if __name__ == "__main__":
    app = QApplication([])

    dialog = ClientDialog()
    dialog.showMaximized()  # 창을 최대화하여 표시

    # Example: Load a sample image to simulate video frames
    sample_image_path = "src/python/test_7.jpg"
    
    if os.path.exists(sample_image_path):
        sample_image = cv2.imread(sample_image_path)
        sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

        # Create a list of sample images
        # frames = [sample_image] * 12

        # Update video frames
        for i in range(12):
            dialog.update_video_frame(sample_image, i)
    else:
        print(f"Sample image not found at {sample_image_path}")

    app.exec()