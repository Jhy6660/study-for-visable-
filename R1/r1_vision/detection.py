"""
目标检测模块 - 封装YOLO推理逻辑
"""
import torch
import numpy as np
from typing import List, Dict, Optional
from ultralytics import YOLO


class ObjectDetector:
    """YOLO目标检测器"""
    
    def __init__(self, model_path: str, device: str = 'cuda', confidence_threshold: float = 0.5):
        """
        初始化检测器
        
        Args:
            model_path: YOLO模型路径
            device: 计算设备 ('cuda' 或 'cpu')
            confidence_threshold: 置信度阈值
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.confidence_threshold = confidence_threshold
        
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            # 开启半精度 (FP16) 加速推理 (仅限 CUDA)
            if self.device.type == 'cuda':
                self.model.model.half()
            print(f'✅ 模型加载成功: {model_path}, 设备: {self.device} (FP16: {self.device.type == "cuda"})')
        except Exception as e:
            raise RuntimeError(f'模型加载失败: {e}')
    
    def detect(self, image: np.ndarray, verbose: bool = False) -> List[Dict]:
        """
        执行目标检测
        
        Args:
            image: BGR格式图像
            verbose: 是否输出详细信息
            
        Returns:
            检测结果列表，每个结果包含 bbox, confidence, class_id
        """
        with torch.no_grad():
            results = self.model(image, verbose=verbose, device=self.device, half=(self.device.type == 'cuda'))
        
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    bbox = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    
                    if confidence >= self.confidence_threshold:
                        detections.append({
                            'bbox': bbox,  # [x1, y1, x2, y2]
                            'confidence': confidence,
                            'class_id': int(box.cls[0]) if hasattr(box, 'cls') else 0
                        })
        
        return detections
    
    def cleanup(self):
        """清理GPU资源"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
