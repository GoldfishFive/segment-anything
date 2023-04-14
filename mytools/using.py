import cv2, os
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

# sam_checkpoint = "sam_vit_h_4b8939.pth"
# sam_checkpoint = "sam_vit_l_0b3195.pth"
sam_checkpoint = "sam_vit_b_01ec64.pth"
device = "cuda"
# model_type = "default" #"vit_h"
# model_type = "vit_l"
model_type = "vit_b"


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def on_EVENT_BUTTONDOWN(event, x, y, flags, param):
    # print(param)
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cv2.circle(param["image_show"], (x, y), 10, (255, 0, 0), thickness=-1)
        cv2.putText(param["image_show"], xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (255, 255, 255), thickness=2)
        cv2.imshow("image", param["image_show"])

        if (param['input_point'].shape[0]==0):
            param['input_point']=np.array([[int(x), int(y)]])
        else:
            param['input_point'] = np.concatenate((param['input_point'], [[int(x), int(y)]]))
            # param['input_point'] = np.append(param['input_point'], np.array([[int(x), int(y)]]))
        param['input_label'] = np.append(param['input_label'], 1)
        print(param['input_point'])
        print(param['input_label'])
        interface(param["predictor"], param['input_point'], param['input_label'], param['image'], param['image_show'])
        #需要重新绘制每一个点，换成点掩码图叠加
        cv2.circle(param["image_show"], (x, y), 10, (255, 0, 0), thickness=-1)
        cv2.putText(param["image_show"], xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (255, 255, 255), thickness=2)
        cv2.imshow("image", param["image_show"])

    if event == cv2.EVENT_RBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cv2.circle(param["image_show"], (x, y), 10, (0, 0, 255), thickness=-1)
        cv2.putText(param["image_show"], xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (255, 255, 255), thickness=2)
        cv2.imshow("image", param["image_show"])
        if (param['input_point'].shape[0]==0):
            param['input_point']=np.array([[int(x), int(y)]])
        else:
            param['input_point'] = np.concatenate((param['input_point'], [[int(x), int(y)]]))
            # param['input_point'] = np.append(param['input_point'], np.array([[int(x), int(y)]]))
        param['input_label'] = np.append(param['input_label'], 0)
        print(param['input_point'])
        print(param['input_label'])
        interface(param["predictor"], param['input_point'], param['input_label'], param['image'], param['image_show'])
        #需要重新绘制每一个点，换成点掩码图叠加
        cv2.circle(param["image_show"], (x, y), 10, (0, 0, 255), thickness=-1)
        cv2.putText(param["image_show"], xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (255, 255, 255), thickness=2)
        cv2.imshow("image", param["image_show"])

def interface(predictor, input_point, input_label, image, image_show):
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        # multimask_output=True,
        multimask_output=False,
    )
    # print(masks.shape)  # (number_of_masks) x H x W
    # print(type(masks))
    # print(np.unique(masks))
    h, w = masks.shape[-2:]
    masks = np.squeeze(masks)
    color_area = np.zeros((h, w, 3), dtype=np.uint8)
    color_area[masks] = [0, 255, 0]
    image_show[masks] = image[masks] * 0.5 + color_area[masks] * 0.5

    print("over!")
    # cv2.imshow("image", image_show)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image_show)
    # plt.show()


if __name__ == "__main__":
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    # image = np.zeros((512, 512, 3), np.uint8)
    image = cv2.imread('notebooks/images/dog.jpg')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_show = image.copy()
    predictor.set_image(image)

    input_point = np.array([])
    input_label = np.array([])
    param = {'image': image, 'predictor': predictor,
             'input_point': input_point, 'input_label': input_label,
             'image_show': image_show}
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_BUTTONDOWN, param=param)

    while (1):
        cv2.imshow("image", image_show)
        if cv2.waitKey(0) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
