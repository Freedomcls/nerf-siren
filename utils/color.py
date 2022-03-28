
import numpy as np
import cv2
import os

def make_atts_map():
    atts =   ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
                'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
    cls_map = {}
    for i, att in enumerate(atts, 1):
        cls_map[i] = att
    return cls_map

def color_cls(img, pred_map, savedir, prefix=""):
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                [255, 0, 85], [255, 0, 170],
                [0, 255, 0], [85, 255, 0], [170, 255, 0],
                [0, 255, 85], [0, 255, 170],
                [0, 0, 255], [85, 0, 255], [170, 0, 255],
                [0, 85, 255], [0, 170, 255],
                [255, 255, 0], [255, 255, 85], [255, 255, 170],
                [255, 0, 255], [255, 85, 255], [255, 170, 255],
                [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    
    vis_im = img.copy().astype(np.uint8)
    vis_pred = pred_map.copy().astype(np.uint8)
    vis_pred = cv2.resize(vis_pred, None, fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
    vis_pred_color = np.zeros((vis_pred.shape[0], vis_pred.shape[1], 3)) + 255
    num_cls = np.max(vis_pred) 
    print(num_cls)
    for i in range(1, num_cls):
        index = np.where(vis_pred==i)
        vis_pred_color[index[0], index[1], :] = part_colors[i]
    vis_pred_color = vis_pred_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_pred_color, 0.6, 0)
    
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    cv2.imwrite(os.path.join(savedir, prefix+"pred_map.png"), vis_pred)
    cv2.imwrite(os.path.join(savedir, prefix+"img_color.png"), vis_im)
    # cv2.imwrite(os.path.join(savedir, prefix+"img_color.png"), cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(savedir, prefix+"img_raw.png"),  cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    # test example
    anno = "./bowen_tou/raw_parse/bowen_tou_00025.png"
    img = "./bowen_tou/images/bowen_tou_00025.jpg"
    img = cv2.imread(img)
    anno = cv2.imread(anno)
    color_cls(img, anno, "./debug")
    
