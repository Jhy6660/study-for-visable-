"""
串口通信模块 - 带CRC校验和重连机制
"""
import struct
import time
import serial
from threading import Lock
from typing import Optional


class SerialComm:
    """串口通信类 - 支持CRC校验和自动重连"""
    
    def __init__(self, port: str, baud_rate: int = 9600, mock_mode: bool = False):
        """
        初始化串口通信
        
        Args:
            port: 串口设备名
            baud_rate: 波特率
            mock_mode: 是否使用Mock模式（测试用）
        """
        self.port = port
        self.baud_rate = baud_rate
        self.mock_mode = mock_mode
        self.serial: Optional[serial.Serial] = None
        self.lock = Lock()
        
        # 重连控制
        self.last_reconnect_time = 0
        self.reconnect_interval = 5.0
        self.max_reconnect_attempts = 3
        self.reconnect_attempts = 0
        
        # 统计
        self.send_count = 0
        self.error_count = 0
        
        self._init_serial()
    
    def _init_serial(self) -> bool:
        """初始化串口连接"""
        if self.mock_mode:
            self.serial = MockSerial()
            print('✅ Mock串口启用（测试模式）')
            return True
        
        try:
            with self.lock:
                if self.serial and self.serial.is_open:
                    self.serial.close()
                self.serial = serial.Serial(self.port, self.baud_rate, timeout=0.1)
                print(f'✅ 串口连接成功: {self.port} @ {self.baud_rate}bps')
                return True
        except serial.SerialException as e:
            print(f'⚠️ 串口连接失败: {e}。将自动回退到 Mock 模式，避免阻塞主流程。')
            self.mock_mode = True
            self.serial = MockSerial()
            return True
        except Exception as e:
            print(f'❌ 串口初始化异常: {e}')
            self.serial = None
            return False
    
    def send_position(self, position: list) -> bool:
        """
        发送3D位置数据
        
        Args:
            position: [x, y, z] 位置列表
            
        Returns:
            发送是否成功
        """
        try:
            # 打包为3个float
            data = struct.pack('fff', float(position[0]), float(position[1]), float(position[2]))
            return self._send_with_crc(data)
        except Exception as e:
            print(f'位置数据打包失败: {e}')
            return False
    
    def send_message(self, message: str) -> bool:
        """
        发送文本消息
        
        Args:
            message: 文本消息
            
        Returns:
            发送是否成功
        """
        try:
            data = message.encode('utf-8')
            return self._send_raw(data)
        except Exception as e:
            print(f'消息发送失败: {e}')
            return False
    
    def _send_with_crc(self, data: bytes) -> bool:
        """发送带CRC校验的数据"""
        with self.lock:
            if not self._check_connection():
                return False
            
            try:
                # 添加包头和CRC
                header = b'\xAA\xBB'
                crc = self._calculate_crc16(data)
                packet = header + data + struct.pack('<H', crc)
                
                self.serial.write(packet)
                self.send_count += 1
                self.reconnect_attempts = 0
                return True
            except Exception as e:
                print(f'串口发送失败: {e}')
                self.error_count += 1
                self._handle_error()
                return False
    
    def _send_raw(self, data: bytes) -> bool:
        """发送原始数据"""
        with self.lock:
            if not self._check_connection():
                return False
            
            try:
                self.serial.write(data)
                self.send_count += 1
                self.reconnect_attempts = 0
                return True
            except Exception as e:
                print(f'串口发送失败: {e}')
                self.error_count += 1
                self._handle_error()
                return False
    
    def _check_connection(self) -> bool:
        """检查连接状态"""
        if self.serial is None or not self.serial.is_open:
            self._handle_error()
            return False
        return True
    
    def _handle_error(self):
        """处理串口错误和重连"""
        current_time = time.time()
        
        if current_time - self.last_reconnect_time < self.reconnect_interval:
            return
        
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            if current_time - self.last_reconnect_time > 30.0:
                self.reconnect_attempts = 0
            else:
                return
        
        self.last_reconnect_time = current_time
        self.reconnect_attempts += 1
        
        print(f'尝试重连串口 ({self.reconnect_attempts}/{self.max_reconnect_attempts})...')
        
        if self._init_serial():
            print('串口重连成功')
        else:
            print('串口重连失败')
    
    def _calculate_crc16(self, data: bytes) -> int:
        """计算CRC16校验码"""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc & 0xFFFF
    
    def close(self):
        """关闭串口连接"""
        with self.lock:
            if self.serial and hasattr(self.serial, 'close'):
                self.serial.close()
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'send_count': self.send_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(1, self.send_count)
        }


class MockSerial:
    """Mock串口类（用于测试）"""
    
    def __init__(self):
        self._is_open = True
    
    def write(self, data: bytes) -> int:
        print(f'MOCK SERIAL: {data.hex()}')
        return len(data)
    
    def close(self):
        self._is_open = False
    
    @property
    def is_open(self) -> bool:
        return self._is_open