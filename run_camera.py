import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
import time
from torchvision.transforms import Compose
from pythonosc import udp_client

# 引入 Depth Anything 模型
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

if __name__ == '__main__':
    
    # --- 优化点 1: 强制使用最小模型 ---
    encoder = 'vits' 
    video_path = 0   

    # OSC 设置
    osc_client = udp_client.SimpleUDPClient("127.0.0.1", 6448)

    # 设备判断
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        print("✅ Running on: CUDA (GTX 950M)")
    else:
        DEVICE = 'cpu'
        print("⚠️ Running on: CPU")
    
    # 加载模型
    print("Loading Model...")
    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE).eval()
    
    # --- 优化点 2: 极致压缩输入分辨率 ---
    # width 从 320 改为 196 (14的倍数)，这是能跑的极限小了
    # 如果觉得太糊，可以试着改回 224 或 252
    target_size = 196 
    
    transform = Compose([
        Resize(
            width=210,
            height=140,
            resize_target=True,
            keep_aspect_ratio=False,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    cap = cv2.VideoCapture(video_path)
    
    # --- 优化点 3: 降低摄像头采集分辨率 ---
    # 强制设为低分辨率，减少 OpenCV 的预处理压力
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    # 字体设置
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # 缓存变量
    last_mean_depth = 0
    
    # --- 优化点 4: 跳帧逻辑变量 ---
    frame_count = 0
    SKIP_FRAMES = 2  # 每隔 2 帧才算一次 (相当于 1/3 的负荷)
    
    # 缓存上一帧的深度结果，跳帧时直接用
    cached_features = [0, 0, 0, 0, 0.5, 0.5] 
    cached_depth_vis = None

    print("🚀 极速模式已启动 (196px | Skip 2 Frames)")

    while cap.isOpened():
        ret, raw_image = cap.read()
        if not ret: break

        frame_count += 1
        
        # --- 核心跳帧逻辑 ---
        # 只有当 frame_count 能被 (SKIP_FRAMES + 1) 整除时，才跑 AI
        if frame_count % (SKIP_FRAMES + 1) == 0:
            
            # 这里不用 resize 了，因为采集时已经设为 320x240 了
            image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
            h, w = image.shape[:2]

            image_tensor = transform({'image': image})['image']
            image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                depth = depth_anything(image_tensor)
            
            # 还原尺寸
            depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
            
            # --- 特征提取 (这一步很快，不需要跳过) ---
            raw_depth = depth.cpu().numpy()
            
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(raw_depth)
            norm_depth = (raw_depth - min_val) / (max_val - min_val + 1e-6)

            mean_depth = float(np.mean(raw_depth))
            occupancy_rate = float(np.sum(norm_depth > 0.7) / (h * w))
            variance = float(np.var(raw_depth))
            
            delta_depth = float(abs(mean_depth - last_mean_depth))
            last_mean_depth = mean_depth 

            focus_x = float(max_loc[0] / w)
            focus_y = float(max_loc[1] / h)

            # 更新缓存
            cached_features = [mean_depth, occupancy_rate, variance, delta_depth, focus_x, focus_y]
            
            # 只有在跑 AI 的这一帧才更新可视化图，节省性能
            depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            cached_depth_vis = depth_vis.cpu().numpy().astype(np.uint8)

        # --- 无论是否跳帧，都发送 OSC (保持 Processing 平滑) ---
        osc_client.send_message("/depth/features", cached_features)
        
        # --- 可视化 (可选：如果还是很卡，把下面这一整段注释掉) ---
        if cached_depth_vis is not None:
            depth_color = cv2.applyColorMap(cached_depth_vis, cv2.COLORMAP_INFERNO)
            # 画个简单的圈标记最近点
            fx = int(cached_features[4] * raw_image.shape[1])
            fy = int(cached_features[5] * raw_image.shape[0])
            cv2.circle(depth_color, (fx, fy), 10, (0, 255, 0), 2)
            
            # 简单拼接，不做文字渲染了，省 CPU
            combined = cv2.hconcat([raw_image, depth_color])
            cv2.imshow('Fast Depth', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()