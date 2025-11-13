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

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)
    else:
        print("using CPU")
    
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

        # test protocol osc communication 
        if 'last_mean_depth' not in locals():
            last_mean_depth = np.mean(raw_depth)
        
        h, w = raw_depth.shape
        center = raw_depth[h//3:2*h//3, w//3:2*w//3]
        left = raw_depth[:, :w//3]
        right = raw_depth[:, -w//3:]
        top = raw_depth[:h//2, :]
        bottom = raw_depth[h//2:, :]

        mean_depth = float(np.mean(raw_depth))
        center_depth = float(np.mean(center))
        left_depth = float(np.mean(left))
        right_depth = float(np.mean(right))
        top_depth = float(np.mean(top))
        bottom_depth = float(np.mean(bottom))
        variance = float(np.var(raw_depth))
        min_depth = float(np.min(raw_depth))
        max_depth = float(np.max(raw_depth))
        delta_depth = float(abs(mean_depth - last_mean_depth))

        last_mean_depth = mean_depth

        depth_features = [
        mean_depth, center_depth, left_depth, right_depth,
        top_depth, bottom_depth, variance, min_depth, max_depth, delta_depth
        ]

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
        