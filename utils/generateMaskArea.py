import os
import cv2
import numpy as np
import torch
from getFrameNumber import get_total_frame_number
from tqdm import tqdm,trange

def generate_mask_area(model,device, save_root,video_path,Atransform):
    total_frame = get_total_frame_number(video_path)
    video_name  = os.path.basename(video_path).split('.')[0]
    save_path = save_root +'/'+ video_name
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Directory " , save_path ,  " Created ")
    model.eval()
    count = 0
    with torch.no_grad():
        video = cv2.VideoCapture(video_path)
        if (video.isOpened()== False):
            print("Error: can't open video file")
        total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if len(os.listdir(save_path)) == total_frame:
            print("Video "+video_name+" is already processed")
            return
        for _ in trange(total_frame, desc="Processing video "+video_name+" "):
        # Capture frame-by-frame
            ret, image = video.read()
            if ret == True:
                original_image = image.copy()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                transformed = Atransform(image=image)
                image = transformed['image']
                image = image.unsqueeze(0).to(device)
                outputs = model(image)

                predicted = torch.argmax(outputs, dim=1)
                mask = predicted.squeeze(0).cpu().numpy().astype(np.float32)
                mask = cv2.resize(mask, (480, 270), interpolation = cv2.INTER_LINEAR)
                mask[mask>0] = 255
                binary_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.uint8)
                extracted_image = cv2.bitwise_and(original_image, binary_mask)

                
                # extracted_image = cv2.cvtColor(extracted_image, cv2.COLOR_RGB2BGR)
                
                image_name = video_name +'_'+str(count)+'.png'
                file_name = save_path +'/'+ image_name
                compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
                cv2.imwrite(file_name,extracted_image,compression_params)
                count += 1
            else:
                break    
    video.release()