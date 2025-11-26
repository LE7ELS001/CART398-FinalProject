import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
import time
from torchvision.transforms import Compose
from tqdm import tqdm
from pythonosc import udp_client


from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


if __name__ == '__main__':

    
    # encoders = ['vits', 'vitb', 'vitl']
    encoder = 'vits'  # default encoder
    video_path = 1

    margin_width = 50
    caption_height = 60
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    
    # initialize OSC client
    osc_client = udp_client.SimpleUDPClient("127.0.0.1", 6448)


    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print("CUDA available:", torch.cuda.is_available())
    # if torch.cuda.is_available():
    #     print("GPU name:", torch.cuda.get_device_name(0))
    #     print("CUDA version:", torch.version.cuda)
    # else:
    #     print("using CPU")
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        print(" Running on: CUDA (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'
        print(" Running on: MPS (Mac Apple Silicon GPU)")
    else:
        DEVICE = 'cpu'
        print(" Running on: CPU (Slow)")

    depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE).eval()
    
    total_params = sum(param.numel() for param in depth_anything.parameters())
    print('Total parameters: {:.2f}M'.format(total_params / 1e6))

    
    transform = Compose([
        Resize(
            width=320,
            height=320,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    # Define the codec and create videoWriter object 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter("output_video.mp4", fourcc, 30.0, (640,480))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open video.")
        exit()


    # test OSC communication
    # for i in range(10):
    #     osc_client.send_message("/wek/inputs", [i * 0.1, 1 - i * 0.1])
    #     print("Sent frame:", i)
    #     time.sleep(0.5)

    while cap.isOpened():
        ret, raw_image = cap.read()

        if  not ret:
            break

        raw_image = cv2.resize(raw_image, (640,480))
        
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        h, w = image.shape[:2]

        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            depth = depth_anything(image)
        
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        
        # raw depth for potential further processing
        raw_depth = depth.cpu().numpy()
        
        # 归一化深度图 (0.0 到 1.0)，方便计算 Occupancy
        # 注意：Depth Anything 输出值越大代表越近 (Disparity)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(raw_depth)
        norm_depth = (raw_depth - min_val) / (max_val - min_val + 1e-6)

        # [Feature 1] Mean Depth (平均深度) -> 代表 "物理压强"
        # 你的身体介入屏幕越深，这个值越大
        mean_depth = float(np.mean(raw_depth))

        # [Feature 2] Occupancy Rate (侵占率/体积感) -> 代表 "淹没感"
        # 计算有多少比例的像素是非常近的 (亮度 > 0.7)
        # 这是一个 3D 概念：不仅仅是靠近，而是"填满"了空间
        occupancy_threshold = 0.7 
        occupancy_rate = float(np.sum(norm_depth > occupancy_threshold) / (h * w))

        # [Feature 3] Variance (方差) -> 代表 "信息复杂度/焦虑"
        # 画面深度层次越多，方差越大 (例如手伸出来，背景很远)
        variance = float(np.var(raw_depth))

        # [Feature 4] Delta Depth (动作幅度/速度) -> 代表 "挣扎"
        # 计算这一帧和上一帧的平均深度差
        delta_depth = float(abs(mean_depth - last_mean_depth))
        last_mean_depth = mean_depth # 更新缓存

        # [Feature 5 & 6] Focus Point (最近点坐标) -> 代表 "对抗焦点"
        # 找到深度最大值的位置 (即离摄像头最近的点，通常是手或头)
        focus_x = float(max_loc[0] / w) # 归一化 0.0-1.0
        focus_y = float(max_loc[1] / h) # 归一化 0.0-1.0

        # --- 3. 发送 OSC 数据 ---
        
        # 顺序必须固定，Wekinator 接收也要按这个顺序
        features = [
            mean_depth,      # Input 1
            occupancy_rate,  # Input 2
            variance,        # Input 3
            delta_depth,     # Input 4
            focus_x,         # Input 5 (直通)
            focus_y          # Input 6 (直通)
        ]

        

        # # test protocol osc communication 
        # if 'last_mean_depth' not in locals():
        #     last_mean_depth = np.mean(raw_depth)
        
        # h, w = raw_depth.shape
        # center = raw_depth[h//3:2*h//3, w//3:2*w//3]
        # left = raw_depth[:, :w//3]
        # right = raw_depth[:, -w//3:]
        # top = raw_depth[:h//2, :]
        # bottom = raw_depth[h//2:, :]

        # mean_depth = float(np.mean(raw_depth))
        # center_depth = float(np.mean(center))
        # left_depth = float(np.mean(left))
        # right_depth = float(np.mean(right))
        # top_depth = float(np.mean(top))
        # bottom_depth = float(np.mean(bottom))
        # variance = float(np.var(raw_depth))
        # min_depth = float(np.min(raw_depth))
        # max_depth = float(np.max(raw_depth))
        # delta_depth = float(abs(mean_depth - last_mean_depth))

        # last_mean_depth = mean_depth

        # depth_features = [
        # mean_depth, center_depth, left_depth, right_depth,
        # top_depth, bottom_depth, variance, min_depth, max_depth, delta_depth
        # ]

        osc_client.send_message("/depth/features", depth_features)
        # print("Sent depth features:", depth_features)

        # Normalize depth for visualization
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0


        
        depth = depth.cpu().numpy().astype(np.uint8)
        depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

        split_region = np.ones((raw_image.shape[0], margin_width, 3), dtype=np.uint8) * 255
        combined_results = cv2.hconcat([raw_image, split_region, depth_color])

        caption_space = np.ones((caption_height, combined_results.shape[1], 3), dtype=np.uint8) * 255
        captions = ['Raw image', 'Depth Anything']
        segment_width = w + margin_width

        for i, caption in enumerate(captions):
                # Calculate text size
                text_size = cv2.getTextSize(caption, font, font_scale, font_thickness)[0]

                # Calculate x-coordinate to center the text
                text_x = int((segment_width * i) + (w - text_size[0]) / 2)

                # Add text caption
                cv2.putText(caption_space, caption, (text_x, 40), font, font_scale, (0, 0, 0), font_thickness)

        final_result = cv2.vconcat([caption_space, combined_results])

        out_video.write(final_result)
        cv2.imshow('Depth Anything - Press q to Exit', final_result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
        