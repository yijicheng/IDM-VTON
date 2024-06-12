import onnxruntime
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import glob
import os
import time
from torch.nn import functional as F
import torch
from torchvision import transforms

# onnxruntime.set_default_logger_severity(3)

def resize_frame_resolution(vision_frame, max_resolution):
	height, width = vision_frame.shape[:2]
	max_width, max_height = max_resolution

	if height > max_height or width > max_width:
		scale = min(max_height / height, max_width / width)
		new_width = int(width * scale)
		new_height = int(height * scale)
		return cv2.resize(vision_frame, (new_width, new_height))
	return vision_frame


def apply_nms(bounding_box_list, iou_threshold):
	keep_indices = []
	dimension_list = np.reshape(bounding_box_list, (-1, 4))
	x1 = dimension_list[:, 0]
	y1 = dimension_list[:, 1]
	x2 = dimension_list[:, 2]
	y2 = dimension_list[:, 3]
	areas = (x2 - x1 + 1) * (y2 - y1 + 1)
	indices = np.arange(len(bounding_box_list))
	while indices.size > 0:
		index = indices[0]
		remain_indices = indices[1:]
		keep_indices.append(index)
		xx1 = np.maximum(x1[index], x1[remain_indices])
		yy1 = np.maximum(y1[index], y1[remain_indices])
		xx2 = np.minimum(x2[index], x2[remain_indices])
		yy2 = np.minimum(y2[index], y2[remain_indices])
		width = np.maximum(0, xx2 - xx1 + 1)
		height = np.maximum(0, yy2 - yy1 + 1)
		iou = width * height / (areas[index] + areas[remain_indices] - width * height)
		indices = indices[np.where(iou <= iou_threshold)[0] + 1]
	return keep_indices

def filter_bbox_by_size(bounding_box_list, size_threshold):
    keep_indices = []
    dimension_list = np.reshape(bounding_box_list, (-1, 4))
    x1 = dimension_list[:, 0]
    y1 = dimension_list[:, 1]
    x2 = dimension_list[:, 2]
    y2 = dimension_list[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for index, area in enumerate(areas):
        if area > size_threshold:
            keep_indices.append(index)

    return keep_indices

def dilate_erode(mask, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilate_mask = cv2.dilate(mask, kernel, iterations=1)
    eroded_mask = cv2.erode(dilate_mask, kernel, iterations=1)
    return eroded_mask


class FaceMaskerV2():
    def __init__(self, detector_model_path="/apdcephfs_cq8/share_1367250/gongyeliu/MyCode/lama_face/models/matting_models/yoloface_8n.onnx", 
                 parser_model_path="/apdcephfs_cq8/share_1367250/gongyeliu/MyCode/lama_face/models/matting_models/face_parser.onnx",
                 seg_model_path="/apdcephfs_cq8/share_1367250/gongyeliu/MyCode/image2video_swapface/data/faceswap_core/.assets/models/XSeg_model.onnx", 
                 face_detector_model_size=640, face_detector_score=0.6, face_parser_model_size=512,
                 device='cuda:0') -> None:
        self.device = device
        if device == 'cpu':
            raise NotImplementedError("CPU execution provider is not supported")
        elif device[:4] == 'cuda':
            try:
                gpu_id = int(device.split(':')[-1])
                execution_provider = (
                    'CUDAExecutionProvider',
                    {
                        'cudnn_conv_algo_search': 'DEFAULT',
                        'device_id': gpu_id
                    }
                )
                # execution_provider = 'CUDAExecutionProvider'
            except:
                raise ValueError(f"Invalid device id: {device}. Please provide a valid device id like 'cuda:0' or 'cuda:1'")
        self.face_detector = onnxruntime.InferenceSession(detector_model_path, providers=[execution_provider])
        self.face_detector_model_size = face_detector_model_size
        self.face_detector_score = face_detector_score
        self.face_parser = onnxruntime.InferenceSession(parser_model_path, providers=[execution_provider])
        self.face_segmentor = onnxruntime.InferenceSession(seg_model_path, providers=[execution_provider])
        self.face_parser_model_size = face_parser_model_size


    def _prepare_detect_frame(self, temp_vision_frame):
        detect_vision_frame = np.zeros((self.face_detector_model_size, self.face_detector_model_size, 3))
        detect_vision_frame[:temp_vision_frame.shape[0], :temp_vision_frame.shape[1]] = temp_vision_frame
        detect_vision_frame = (detect_vision_frame - 127.5) / 128.0
        detect_vision_frame = np.expand_dims(detect_vision_frame.transpose(2, 0, 1), axis = 0).astype(np.float32)
        return detect_vision_frame
    
    def detect_faces(self, vision_frame):
        temp_vision_frame = resize_frame_resolution(vision_frame, (self.face_detector_model_size, self.face_detector_model_size))
        ratio_height = vision_frame.shape[0] / temp_vision_frame.shape[0]
        ratio_width = vision_frame.shape[1] / temp_vision_frame.shape[1]
        bbox_list = []
        face_lmk_5_list = []
        score_list = []
        
        # save image
        # cv2.imwrite("detect_input.png", vision_frame)

        detect_vision_frame = self._prepare_detect_frame(temp_vision_frame)

        detections = self.face_detector.run(None, {"images": detect_vision_frame})[0]
        detections = np.squeeze(detections).T
        bbox_raw, score_raw, face_lmk_5_raw = np.split(detections, [4, 5], axis = 1)
        keep_indices = np.where(score_raw > self.face_detector_score)[0]

        if keep_indices.any():
            bbox_raw, face_lmk_5_raw, score_raw = bbox_raw[keep_indices], face_lmk_5_raw[keep_indices], score_raw[keep_indices]
            for bbox in bbox_raw:
                bbox_list.append(np.array(
                [
                    (bbox[0] - bbox[2] / 2) * ratio_width,
                    (bbox[1] - bbox[3] / 2) * ratio_height,
                    (bbox[0] + bbox[2] / 2) * ratio_width,
                    (bbox[1] + bbox[3] / 2) * ratio_height
                ]))
            face_lmk_5_raw[:, 0::3] = (face_lmk_5_raw[:, 0::3]) * ratio_width
            face_lmk_5_raw[:, 1::3] = (face_lmk_5_raw[:, 1::3]) * ratio_height
            for face_lmk_5 in face_lmk_5_raw:
                face_lmk_5_list.append(np.array(face_lmk_5.reshape(-1, 3)[:, :2]))
            score_list = score_raw.ravel().tolist()

        keep_indices1 = apply_nms(bbox_list, 0.1)
        keep_indices2 = filter_bbox_by_size(bbox_list, 1024) # at least 32x32 face(for resized 640x640)
        keep_indices = list(set(keep_indices1) & set(keep_indices2))

        if len(keep_indices) == 0:
            return [], [], []

        bbox_list = [bbox_list[i] for i in keep_indices]
        face_lmk_5_list = [face_lmk_5_list[i] for i in keep_indices]
        score_list = [score_list[i] for i in keep_indices]
        
        if len(keep_indices) == 0:
            return [], [], []
        # sort the faces by score(from high to low)
        bbox_list, face_lmk_5_list, score_list = zip(*sorted(zip(bbox_list, face_lmk_5_list, score_list), reverse=True, key=lambda x: x[2]))

        return bbox_list, face_lmk_5_list, score_list

    def create_face_mask(self, vision_frame, face_bbox=None):
        if face_bbox is not None:
            face_bbox = [int(i) for i in face_bbox]
            crop_vision_frame = vision_frame[int(face_bbox[1]):int(face_bbox[3]), int(face_bbox[0]):int(face_bbox[2])]
        else:
            crop_vision_frame = vision_frame
        
        input_size = crop_vision_frame.shape[:2]
        prepare_vision_frame = cv2.resize(crop_vision_frame, (self.face_parser_model_size, self.face_parser_model_size))
        prepare_vision_frame = np.expand_dims(prepare_vision_frame, axis=0).astype(np.float32) / 127.5 - 1.0
        prepare_vision_frame = prepare_vision_frame.transpose(0, 3, 1, 2)        
        face_mask_from_seg = self.face_segmentor.run(None,
        {
            self.face_segmentor.get_inputs()[0].name: prepare_vision_frame
        })[0][0]

        face_mask_from_seg = (face_mask_from_seg[:, :, 0] > 0.5).astype(np.float32)
        face_mask_from_seg = face_mask_from_seg[:, :, np.newaxis]

        face_mask = dilate_erode(face_mask_from_seg)

        face_mask = ((face_mask > 0.5) * 255).astype(np.uint8)

        return face_mask
    
    @torch.no_grad()
    def get_face(self, images):
        '''
        Input:
            images: tensor, shape = [B, C, H, W]
        Output:
            face: tensor, shape = [B, C, H, W]
            bg: tensor, shape = [B, C, H, W]
        '''
        b, c, h, w = images.shape
        # print(images.shape)
        # extract mask
        images_resize = F.interpolate(images, (256, 256), mode='bilinear', align_corners=True)
        images_np = images_resize.cpu().numpy()
        images_np = images_np.transpose(0, 2, 3, 1)


        face_masks_from_seg = self.face_segmentor.run(None,
        {
            self.face_segmentor.get_inputs()[0].name: images_np
        })[0]

        face_masks_from_seg = (face_masks_from_seg[:, :, :, 0] > 0.5).astype(np.float32)
        face_masks_from_seg = face_masks_from_seg[:, np.newaxis, :, :]

        face_masks = np.concatenate([dilate_erode(face_mask)[np.newaxis, :, :, :] for face_mask in face_masks_from_seg], axis=0)
        face_masks = (face_masks > 0.5).astype(np.float32)

        face_masks = torch.from_numpy(face_masks).to(images.device)

        # print(face_masks.shape)
        face_masks = F.interpolate(face_masks, (h, w), mode='nearest')

        face = images * face_masks
        bg = images * (1 - face_masks)

        return face, bg


def expand_bbox(bbox, img_shape, scale=[1.4, 1.5, 1.4, 1.3]):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1

    left_margin, top_margin, right_margin, bottom_margin = h * scale[0] - h, h * scale[1] - h, w * scale[2] - w, w * scale[3] - w
    # adjust the margin to make the bbox square
    max_side = max(w + left_margin + right_margin, h + top_margin + bottom_margin)
    left_margin = (max_side - w) / (left_margin + right_margin + 1e-6) * left_margin
    top_margin = (max_side - h) / (top_margin + bottom_margin + 1e-6) * top_margin
    right_margin = (max_side - w) / (left_margin + right_margin + 1e-6) * right_margin
    bottom_margin = (max_side - h) / (top_margin + bottom_margin + 1e-6) * bottom_margin

    x1_expand = int(max(cx - w / 2 - left_margin, 0))
    y1_expand = int(max(cy - h / 2 - top_margin, 0))
    x2_expand = int(min(cx + w / 2 + right_margin, img_shape[1]))
    y2_expand = int(min(cy + h / 2 + bottom_margin, img_shape[0]))

    # generate a bbox mask of input bbox (based on the output bbox)
    box_mask = np.zeros((y2_expand - y1_expand, x2_expand - x1_expand, 1), dtype=np.uint8)
    box_mask[y1 - y1_expand:y2 - y1_expand, x1 - x1_expand:x2 - x1_expand, :] = 255

    return [x1_expand, y1_expand, x2_expand, y2_expand], box_mask


def get_face_mask(image, face_masker):
    # image: h w c # RGB numpy
    # face_masker = FaceMaskerV2(device=device)
    bbox_list, face_lmk_5_list, score_list = face_masker.detect_faces(image)
    face_bbox, box_mask = expand_bbox(bbox_list[0], (image.shape[1], image.shape[0]))

    face_image = image[int(face_bbox[1]):int(face_bbox[3]), int(face_bbox[0]):int(face_bbox[2])]


    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_image = transforms.ToTensor()(face_image)
    face_image = face_image.unsqueeze(0).to(face_masker.device)

    face_mask, bg_mask = face_masker.get_face(face_image) # 1 c h w
    face_mask = face_mask.squeeze(0).cpu().numpy()
    face_mask = face_mask.transpose(1, 2, 0)
    face_mask = (face_mask * 255).astype(np.uint8) # h w c

    mask = np.zeros_like(image)
    mask[int(face_bbox[1]):int(face_bbox[3]), int(face_bbox[0]):int(face_bbox[2])] = face_mask
    mask = mask != 0 # bool: h w c
    # mask = (mask * 255).astype(np.uint8)
    # print(mask.shape)
    
    return mask

if __name__ == '__main__':
     
    device = 'cuda:0'

    face_masker = FaceMaskerV2(device=device)


    # img_paths = ["/apdcephfs_cq8/share_1367250/gongyeliu/MyCode/lama_face/test.jpg"]
    img_paths = ["/apdcephfs_cq8/share_1367250/gongyeliu/MyCode/lama_face/test_zhangwei.jpg"]
    out_dir = "./lama_face/"
    os.makedirs(out_dir, exist_ok=True)
    for img_path in tqdm(img_paths):
        image_np = cv2.imread(img_path)
        # print(image_np.shape) # h w c

        bbox_list, face_lmk_5_list, score_list = face_masker.detect_faces(image_np[:, :, ::-1])
        face_bbox = bbox_list[0]
        # print(face_bbox)
        face_bbox, box_mask = expand_bbox(bbox_list[0], (image_np.shape[1], image_np.shape[0]))
        # print(face_bbox)

        face_image = image_np[int(face_bbox[1]):int(face_bbox[3]), int(face_bbox[0]):int(face_bbox[2])]
        # face_image = cv2.resize(face_image, (512, 512))

        # to tensor
        image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        image = transforms.ToTensor()(image)
        image = image.unsqueeze(0).to(device)

        face, bg = face_masker.get_face(image) # 1 c h w
        # print(face.shape, bg.shape)

        # save
        out_path = f"{out_dir}/{img_path.split('/')[-1].split('.')[0]}"



        face = face.squeeze(0).cpu().numpy()
        face = face.transpose(1, 2, 0)
        face = (face * 255).astype(np.uint8)
        face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)


        face = cv2.addWeighted(face, 0.5, face_image, 0.5, 0)
        cv2.imwrite(f"{out_path}_face.png", face)

        image_np[:, :] = 0
        image_np[int(face_bbox[1]):int(face_bbox[3]), int(face_bbox[0]):int(face_bbox[2])] = face
        mask = image_np != 0
        mask = (mask * 255).astype(np.uint8)
        print(mask.shape)



        cv2.imwrite(f"{out_path}_image.png", image_np)
        cv2.imwrite(f"{out_path}_mask.png", mask)

        bg = bg.squeeze(0).cpu().numpy()
        bg = bg.transpose(1, 2, 0)
        bg = (bg * 255).astype(np.uint8)
        bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)

        bg = cv2.addWeighted(bg, 0.5, face_image, 0.5, 0)
        cv2.imwrite(f"{out_path}_bg.png", bg)
        
