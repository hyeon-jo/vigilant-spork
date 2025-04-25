# -*- coding: utf-8 -*-
from PySide6.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLineEdit, QLabel, QGridLayout, QWidget, QSizePolicy, QFrame,
                             QStatusBar, QProgressBar, QGroupBox, QSplitter)
from PySide6.QtGui import QImage, QPixmap, QFont, QColor, QPalette
from PySide6.QtCore import Qt, Signal, Slot, QMutex, QTimer, QSize

import cv2
import os
import socket
import struct
import threading
import time
import numpy as np

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
        self.setFixedSize(20, 20)
        self.set_status("disconnected")
        self.setToolTip("Server Connection Status")

    def set_status(self, status):
        if status == "connected":
            self.setStyleSheet("""
                QLabel {
                    background-color: #2ecc71;
                    border-radius: 10px;
                    border: 2px solid #27ae60;
                }
            """)
        elif status == "disconnected":
            self.setStyleSheet("""
                QLabel {
                    background-color: #e74c3c;
                    border-radius: 10px;
                    border: 2px solid #c0392b;
                }
            """)
        else:
            self.setStyleSheet("""
                QLabel {
                    background-color: #f1c40f;
                    border-radius: 10px;
                    border: 2px solid #f39c12;
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
        
        server_layout.addWidget(server1_label, 0, 0)
        server_layout.addWidget(self.server1_ip_edit, 0, 1)
        server_layout.addWidget(self.server1_status_indicator, 0, 2)
        server_layout.addWidget(self.server1_status_indicator_log, 0, 3)
        server_layout.addWidget(server2_label, 1, 0)
        server_layout.addWidget(self.server2_ip_edit, 1, 1)
        server_layout.addWidget(self.server2_status_indicator, 1, 2)
        server_layout.addWidget(self.server2_status_indicator_log, 1, 3)
        server_group.setLayout(server_layout)
        
        # 버튼 그룹
        button_group = QGroupBox("Control")
        button_layout = QHBoxLayout()
        
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
        
        # 레이아웃 조립
        main_layout.addWidget(control_panel)
        main_layout.addWidget(video_group)
        main_layout.addWidget(lidar_group)
        
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
        self.server1_socket_send_thread = None
        self.server1_socket_receive_thread = None
        self.server2_socket_send_thread = None
        self.server2_socket_receive_thread = None
        
        # 테마 적용
        self.apply_modern_theme()

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
            self.send_logging_signal(message_type=20)
            self.send_logging_signal(message_type=19)

    def custom_close(self):
        # Perform any cleanup or additional actions here
        if self.is_logging:
            self.stop_logging()  # Assuming you have a method to stop logging

        if self.server1_socket:
            self.server1_socket.close()  # Close server1 socket if it exists

        if self.server2_socket:
            self.server2_socket.close()  # Close server2 socket if it exists

        
        if self.server1_socket_log:
            self.server1_socket_log.close()  # Close server1 socket if it exists

        if self.server2_socket_log:
            self.server2_socket_log.close()  # Close server2 socket if it exists

        self.ip1_status = -1
        self.ip2_status = -1

        print("wait for close ... 3s")
        time.sleep(3)

        if self.server1_socket_send_thread != None:
            self.server1_socket_send_thread.join()
        if self.server2_socket_send_thread != None:
            self.server2_socket_send_thread.join()

        if self.server1_socket_receive_thread != None:
            self.server1_socket_receive_thread.join()
        if self.server2_socket_receive_thread != None:
            self.server2_socket_receive_thread.join()


        self.close()  # Call the original close method to close the dialog
        

    def closeEvent(self, event):
        self.custom_close()

        # Accept the close event to actually close the dialog
        event.accept()


    def stop_logging(self):
        print()

    @Slot()
    def on_connect_clicked(self):
        ip1 = self.server1_ip_edit.text()
        ip2 = self.server2_ip_edit.text()

        port = 9090
        log_port = 9091

        # 서버 연결 시도
        time.sleep(0.5)
        threading.Thread(target=self.connect_to_server, args=(ip1, port, self.server1_status_indicator, 1)).start()
        time.sleep(0.5)
        threading.Thread(target=self.connect_to_server, args=(ip2, port, self.server2_status_indicator, 2)).start()
        time.sleep(0.5)
        threading.Thread(target=self.connect_to_server, args=(ip1, log_port, self.server1_status_indicator_log, 3)).start()
        # time.sleep(0.5)
        # threading.Thread(target=self.connect_to_server, args=(ip2, log_port, self.server2_status_indicator_log, 4)).start()

    def connect_to_server(self, ip, port, status_indicator, server_id):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((ip, port))
            status_indicator.set_status("connected")
            print(f"Successfully connected to Server: {ip} {port}")

            if server_id == 1:
                self.server1_socket = sock
                self.server1_socket_receive_thread = threading.Thread(target=self.receive_data, args=(sock, 1))
                self.server1_socket_receive_thread.start()
                self.server1_socket_send_thread = threading.Thread(target=self.send_data, args=(sock, 1))
                self.server1_socket_send_thread.start()
            elif server_id == 2:
                self.server2_socket = sock
                self.server2_socket_receive_thread = threading.Thread(target=self.receive_data, args=(sock, 2))
                self.server2_socket_receive_thread.start()
                self.server2_socket_send_thread = threading.Thread(target=self.send_data, args=(sock, 2))
                self.server2_socket_send_thread.start()

            elif server_id == 3:
                self.server1_socket_log = sock
                # self.send_to_start_logging(sock)
                # self.server1_socket_receive_log_thread = threading.Thread(target=self.receive_data_log, args=(sock, 1))
                # self.server1_socket_receive_log_thread.start()
                # self.server1_socket_send_log_thread = threading.Thread(target=self.send_data_log, args=(sock, 1))
                # self.server1_socket_send_log_thread.start()

            elif server_id == 4:
                self.server2_socket_log = sock
                # threading.Thread(target=self.receive_data, args=(sock, 4)).start()
                # threading.Thread(target=self.send_data, args=(sock, 4)).start()

        except Exception as e:
            status_indicator.set_status("disconnected")
            print(f"Failed to connect to Server: {ip}. Error: {e}")

    def send_LINK_REQ(self, sock):
        current_time = int(time.time())  # Current timestamp in milliseconds

        header = Protocol_Header(
            time_stamp=current_time, 
            message_type=1,     # LINK_REQ
            sequence_number=0, 
            body_length=1,
            mResult=0)
        print(current_time)

        # 헤더 직렬화
        serialized_data = header.serialize()

        # 직렬화된 데이터 보내기
        sock.sendall(serialized_data)
        
        print('Header sent to client:', header.serialize())

    def send_REC_INFO_REQ(self, sock):
        current_time = int(time.time())  # Current timestamp in milliseconds
        header = Protocol_Header(
            time_stamp=current_time, 
            message_type=3,     # REC_INFO_REQ
            sequence_number=0, 
            body_length=1,
            mResult=0)
        print(current_time)

        # 헤더 직렬화
        serialized_data = header.serialize()

        # 직렬화된 데이터 보내기
        sock.sendall(serialized_data)
        
        print('REC_INFO_REQ Header sent to client:', header.serialize())


    def send_SEND_REQ(self, sock):

        current_time = int(time.time())

        # send DATA_SEND_REQ
        header = DATA_SEND_REQ_Header(
            time_stamp=current_time, 
            message_type=17,
            sequence_number=1, 
            body_length=8,
            request_status = 0,
            data_type = 1,
            sensor_channel = 4294967295,
            service_id = 0, 
            network_id = 0)

        print(current_time)
        # 헤더 직렬화
        serialized_data = header.serialize()
        # 직렬화된 데이터 보내기
        sock.sendall(serialized_data)
        print('DATA_SEND_REQ_Header sent to client:', header.serialize())


    def send_data(self, sock, server_id):
        try:
            while sock:
                # message = f"Data from Server {server_id}"
                # sock.sendall(message.encode('utf-8'))
                # time.sleep(1)  # 1초마다 데이터 보내기
                
                self.mutex_send.lock()

                time.sleep(0.1)

                if server_id == 1:
                    if self.ip1_status == 0:
                        self.send_LINK_REQ(sock)
                        self.ip1_status = 1

                    if self.ip1_status == 2:
                        self.send_REC_INFO_REQ(sock)
                        self.ip1_status = 3
                        self.send_SEND_REQ(sock)

                if server_id == 2:
                    if self.ip2_status == 0:
                        self.send_LINK_REQ(sock)
                        self.ip2_status = 1

                    if self.ip2_status == 2:
                        self.send_REC_INFO_REQ(sock)
                        self.ip2_status = 3
                        self.send_SEND_REQ(sock)

                if server_id == 3:
                    print("send id 3")

                if server_id == 4:
                    print("send id 4")

                self.mutex_send.unlock()

                if self.ip1_status == -1 or self.ip2_status == -1:
                    return

        except Exception as e:
            print(f"Failed to send data to Server {server_id}. Error: {e}")
            if server_id == 1:
                self.server1_status_indicator.set_status("disconnected")
                self.server1_socket = None
            elif server_id == 2:
                self.server2_status_indicator.set_status("disconnected")
                self.server2_socket = None

        finally:
            sock.close()
            print(f"Connection closed with Server {server_id}")
            if server_id == 1:
                self.server1_status_indicator.set_status("disconnected")
                self.server1_socket = None
            elif server_id == 2:
                self.server2_status_indicator.set_status("disconnected")
                self.server2_socket = None
       
            # elif server_id == 3:
            #     self.server1_status_indicator_log.set_status("disconnected")
            #     self.server1_socket_log = None
            # elif server_id == 4:
            #     self.server2_status_indicator_log.set_status("disconnected")
            #     self.server2_socket_log = None


    def receive_LINE_ACK(self, sock):
        # Receive the header from the server
        response_data = sock.recv(struct.calcsize("<QBQIB"))
        received_size = len(response_data)
        print("!!!!!! received size: ", received_size)

        if response_data:
            # 응답 데이터 역직렬화
            response_header = Protocol_Header.deserialize(response_data)
            print(f"받은 헤더: TimeStamp={response_header[0]}, MessageType={response_header[1]}, SequenceNumber={response_header[2]}, BodyLength={response_header[3]}, mResult={response_header[4]}")
        else:
            print("응답 데이터를 받지 못했습니다.")

    def receive_SENSOR_DATA(self, sock):
        # Receive the header from the server
        response_data = sock.recv(22)

        received_size = len(response_data)
        # print("!!!!!! received size: ", received_size)

        fmt = "<QBQIB"
        # dese = struct.unpack(fmt, response_data)
        dese = struct.unpack(fmt, response_data[0:22])
        # print(dese)

        recv_data = b""
        data_tmp = b""
        while True:
            if (dese[3] - 1) - len(recv_data) < 1024:
                data_tmp = sock.recv((dese[3] - 1) - len(recv_data))
                # print("remained data size: ", len(data_tmp), (dese[3] - 1) - len(recv_data))
                recv_data += data_tmp
                break

            data_tmp = sock.recv(1024)
            # print("!! remained data size: ", len(data_tmp), (dese[3] - 1) - len(recv_data))
            recv_data += data_tmp
            
        # print(" recv_data total received size: ", len(recv_data))

        if dese[1] == 5:

            # 구조체 정의
            SEND_DATA_HEADER_FORMAT = "IIQBBHHBBII"
            SEND_DATA_HEADER_SIZE = struct.calcsize(SEND_DATA_HEADER_FORMAT)

            # 앞에 2개의 header 가 더 있지만, II 를 빼고 7 만큼 offset 해야지만 정상 출력이 가능

            offset = 7
        
            # 헤더 파싱
            header = struct.unpack(SEND_DATA_HEADER_FORMAT, recv_data[offset:SEND_DATA_HEADER_SIZE+offset])
            
            send_data_header =  {
                "mCurrentNumber": header[0],
                "mFrameNumber": header[1],
                "mTimestamp": header[2],
                "mSensorType": header[3],
                "mChannel": header[4],
                "mImgWidth": header[5],
                "mImgHeight": header[6],
                "mImgDepth": header[7],
                "mImgFormat": header[8],
                "mNumPoints": header[9],
                "mPayloadSize": header[10]
            }
            # print(send_data_header)
            
            # pre_data = client_socket.recv(30)
            # data = client_socket.recv(1024)
            if send_data_header['mSensorType'] == 1:    # Sensor: CAMERA
                    
                image_buffer = recv_data[SEND_DATA_HEADER_SIZE+offset:]

                # 광각 영상에 대해서만 처리
                # height = 614
                # width = 768
                
                grayscale_image = self.yuv422uyvy_to_grayscale(image_buffer, send_data_header['mImgWidth'], send_data_header['mImgHeight'])
                # cv2.imwrite(str("src/python/test_grayscale_" + str(header[4]) + ".jpg"), grayscale_image)
                # cv2.imshow(str("test" + str(header[4])), grayscale_image)
                # cv2.waitKey(10)
                # print(str("received image data: ch " + str(header[4])))
                self.update_video_frame(grayscale_image, send_data_header['mChannel'])


            if send_data_header['mSensorType'] == 2:    # Sensor: LIDAR
                point_cloud_buffer = recv_data[SEND_DATA_HEADER_SIZE+offset:]
                print("point_cloud_buffer length", point_cloud_buffer.__len__())
                print("mNumPoints", send_data_header['mNumPoints'])

                point_cloud_data = np.frombuffer(point_cloud_buffer, dtype=np.float32).reshape((send_data_header['mNumPoints'], 4))
                # print(point_cloud_data[0:10])


    def receive_data(self, sock, server_id):
        try:
            while sock:
                
                self.mutex_receive.lock()

                time.sleep(0.1)

                if server_id == 1:
                    if self.ip1_status == 1:
                        self.receive_LINE_ACK(sock)
                        self.ip1_status = 2
                                
                    if self.ip1_status == 3:
                        self.receive_SENSOR_DATA(sock)

                if server_id == 2:
                    if self.ip2_status == 1:
                        self.receive_LINE_ACK(sock)
                        self.ip2_status = 2
                                
                    if self.ip2_status == 3:
                        self.receive_SENSOR_DATA(sock)

                if server_id == 3:
                    print("receive id: 3")

                if server_id == 4:
                    print("receive id: 4")

                self.mutex_receive.unlock()

                if self.ip1_status == -1 or self.ip2_status == -1:
                    return
                        
        except Exception as e:
            print(f"Failed to receive data from Server {server_id}. Error: {e}")
        finally:
            sock.close()
            print(f"Connection closed with Server {server_id}")
            if server_id == 1:
                self.server1_status_indicator.set_status("disconnected")
                self.server1_socket = None
            elif server_id == 2:
                self.server2_status_indicator.set_status("disconnected")
                self.server2_socket = None
            
            # elif server_id == 3:
            #     self.server1_status_indicator_log.set_status("disconnected")
            #     self.server1_socket_log = None
            # elif server_id == 4:
            #     self.server2_status_indicator_log.set_status("disconnected")
            #     self.server2_socket_log = None


    def send_logging_signal(self, message_type=19):
        try:
            print(f"start logging {self.timestamp}")

            # current_time = int(time.time_ns())  # Current timestamp in milliseconds
            # self.timestamp = current_time

            # HADF_LOGGING_START_CONT_REQ 19

            header = Protocol_LoggingInfo(
                time_stamp=self.timestamp, 
                message_type=message_type,     # Logging start
                sequence_number=0, 
                body_length=84,
                time_stamp_log_start=self.timestamp if message_type == 19 else 0,
                time_stamp_log_end=self.timestamp if message_type == 20 else 0,
                metasize=64,
                metadescription=str("aaaaaa").encode('utf-8')
                )

            # print(self.timestamp)

            # 헤더 직렬화
            serialized_data = header.serialize()

            # 직렬화된 데이터 보내기
            if self.server1_socket_log:
                self.server1_socket_log.sendall(serialized_data)

            if self.server2_socket_log:
                self.server2_socket_log.sendall(serialized_data)
            
            # print('Header sent to client:', header.serialize())

            # receive

            # Receive the header from the server
            response_data = sock.recv(struct.calcsize("<QBQI"))
            received_size = len(response_data)
            # print("!!!!!! received size: ", received_size)

            if response_data:
                # 응답 데이터 역직렬화
                response_header = Protocol_Header_org.deserialize(response_data)
                # print(f"받은 헤더: TimeStamp={response_header[0]}, MessageType={response_header[1]}, SequenceNumber={response_header[2]}, BodyLength={response_header[3]}, mResult={response_header[4]}")
            else:
                print("응답 데이터를 받지 못했습니다.")

            
        except Exception as e:
            print(f"Failed to connect to Server: . Error: {e}")


    @Slot()
    def on_logging_start_clicked(self):
        if not self.is_logging:
            self.logging_start_button.setText("Stop Logging")
            self.is_logging = True
            self.timestamp = int(time.time_ns())
            
            self.send_logging_signal(message_type=19)
            self.timer.start(29900)
        else:
            self.logging_start_button.setText("Start Logging")
            self.timer.stop()
            self.send_logging_signal(message_type=20)
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