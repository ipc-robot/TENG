import sys
import socket
import numpy as np
import time
from datetime import datetime
from multiprocessing import Process, Queue
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5 import QtWidgets, QtCore

# 显著增加系统socket缓冲区大小
SOCKET_BUFFER_SIZE = 1024 * 1024  # 1MB

class UDPDataCollector:
    def __init__(self, ip, port, buffer_size=4096, queue=None):
        self.ip = ip
        self.port = port
        self.buffer_size = buffer_size
        self.data_buffer = b''  # 数据缓冲区
        self.running = False
        self.sock = None
        self.queue = queue
        self.processed_data = []
        
        # 定义数据包标识
        self.START_SEQ = bytes([0x5A, 0x5A, 0x5A, 0x5A])
        self.END_SEQ = bytes([0x0D, 0x0A, 0x0D, 0x0A])

        # 统计信息
        self.stats = {
            'received_packets': 0,
            'processed_packets': 0,
            'start_time': None
        }

    def start(self, collection_time=10):
        self.running = True
        self.stats['start_time'] = time.time()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        
        # 设置更大的接收缓冲区
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_BUFFER_SIZE)
        
        end_time = time.time() + collection_time
        
        while time.time() < end_time:
            try:
                data, _ = self.sock.recvfrom(self.buffer_size)
                self.stats['received_packets'] += 1
                self._process_packet(data)
                self.stats['processed_packets'] += 1
            except Exception as e:
                print(f"处理数据错误: {e}")
                    
        self.stop()
        self.save_data()

    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()

    def _process_packet(self, data):
        """处理单个数据包"""
        self.data_buffer += data
        
        segments = self.find_all_segments(self.data_buffer, self.START_SEQ, self.END_SEQ)
        
        processed_data = []
        for segment in segments:
            result = self.process_segment(segment[12:-4])  # 得到一个形状为(10,32)的矩阵
            if result is not None:
                processed_data.append(result)
        
        if processed_data:
            final_array = np.vstack(processed_data)
            self.processed_data.append(final_array)  # 添加到全局的结果中
            if self.queue is not None:
                self.queue.put(final_array)          # 通过队列传递给QT，如果QT存在的话。
        
        # 保留未处理的部分缓冲区
        if segments:
            last_end = self.data_buffer.rfind(self.END_SEQ)
            if last_end != -1:
                self.data_buffer = self.data_buffer[last_end + len(self.END_SEQ):]
        
        if len(self.data_buffer) > 10000:
            self.data_buffer = b''

    def process_segment(self,bag_segment):
        """处理单个数据段，返回 (10,32) 的数组"""
        try:
            if len(bag_segment) != 680:
                return None
                
            i16_array = np.frombuffer(bag_segment, dtype='<i2')            
            float_array_1_16 = (i16_array.reshape(20,-1)/3276.8)[:10,:-1]
            float_array_17_32 = (i16_array.reshape(20,-1)/3276.8)[10:,:-1]
            combined_array = np.hstack((float_array_1_16, float_array_17_32))

            return combined_array
            
        except Exception as e:
            print(f"处理数据段时出错: {e}")
            return None

    def find_all_segments(self, data, start_seq, end_seq):
        """查找所有符合条件的数据段"""
        segments = []
        position = 0
        
        while True:
            start_index = data.find(start_seq, position)
            if start_index == -1:
                break
            
            end_index = data.find(end_seq, start_index)
            if end_index == -1:
                break
            
            segment = data[start_index:end_index + len(end_seq)]

            if len(segment) == 696:
                segments.append(segment)
            
            position = end_index + 4
            
        return segments

    def save_data(self):
        """保存处理后的数据"""
        if self.processed_data:
            try:
                # 合并所有处理后的数据
                final_array = np.vstack(self.processed_data)
                print(final_array)
                
                # 创建带时间戳的文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f'processed_data_{timestamp}.csv'
                
                # 保存为CSV文件
                np.savetxt(filename, final_array, delimiter=',',fmt='%.10f')
                
                # 打印统计信息
                elapsed_time = time.time() - self.stats['start_time']
                print("\n数据采集统计:")
                print(f"运行时间: {elapsed_time:.2f} 秒")
                print(f"接收包数: {self.stats['received_packets']}")
                print(f"处理包数: {self.stats['processed_packets']}")
                print(f"有效数据段数: {len(self.processed_data)}")
                print(f"最终数据形状: {final_array.shape}")
                print(f"数据已保存到文件: {filename}")
                
            except Exception as e:
                print(f"保存数据时出错: {e}")
        else:
            print("没有可保存的数据")

class RealTimePlotter(QtWidgets.QMainWindow):
    def __init__(self, queue, buffer_size=100):
        super().__init__()
        self.queue = queue
        self.initUI()
        self.buffer_size = buffer_size

        # Initialize data storage
        self.data = np.zeros((self.buffer_size, 32))  # 32 channels, each with 'buffer_size' points

    def initUI(self):
        self.setWindowTitle('Real-Time Data Plotting')
        self.graphWidget = pg.GraphicsLayoutWidget(show=True)
        self.setCentralWidget(self.graphWidget)

        # Create 32 plots
        self.plots = [self.graphWidget.addPlot(row=i//8, col=i%8) for i in range(32)]
        for plot in self.plots:
            plot.setYRange(-1, 1)  # Set y-axis range to [-1, 1]
        self.curves = [plot.plot([], []) for plot in self.plots]

        # Set up a timer for periodic updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(10)


    def update(self):
        # Check if there's new data in the queue
        if not self.queue.empty():
            # Retrieve new data from the queue
            final_array = self.queue.get()
            num_new_points = final_array.shape[0]

            # Ensure that the new data does not exceed buffer size
            if num_new_points > self.buffer_size:
                final_array = final_array[-self.buffer_size:]  # Take only the most recent part

            # Shift old data and append new data
            self.data = np.roll(self.data, -num_new_points, axis=0)
            self.data[-num_new_points:] = final_array

            # Update each curve with the new data using numpy indexing
            for i, curve in enumerate(self.curves):
                curve.setData(self.data[:, i])


def start_udp_collector(queue):
    collector = UDPDataCollector("192.168.4.2", 8080, buffer_size=4096, queue=queue)
    collector.start(collection_time=4000000)


def start_qt_plotter(queue):
    app = QtWidgets.QApplication(sys.argv)
    plotter = RealTimePlotter(queue,buffer_size=1000)
    plotter.show()
    sys.exit(app.exec_())


def main():
    queue = Queue()

    udp_process = Process(target=start_udp_collector, args=(queue,))
    udp_process.start()

    qt_process = Process(target=start_qt_plotter, args=(queue,))
    qt_process.start()

    udp_process.join()
    qt_process.join()


if __name__ == "__main__":
    main()
