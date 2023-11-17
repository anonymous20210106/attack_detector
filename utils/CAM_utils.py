import cv2
import torch
import torchvision
import numpy as np

COLORS = np.random.uniform(0, 255, size=(80, 3))

class YOLOBoxScoreTarget:
    def __init__(self, labels, bounding_boxes, iou_threshold=0.5):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.iou_threshold = iou_threshold

    def __call__(self, model_outputs):
        output = torch.Tensor([0])
        if torch.cuda.is_available():
            output = output.cuda()

        if len(model_outputs["boxes"]) == 0:
            return output

        for box, label in zip(self.bounding_boxes, self.labels):
            box = torch.Tensor(box[None, :])
            if torch.cuda.is_available():
                box = box.cuda()

            ious = torchvision.ops.box_iou(box, model_outputs["boxes"])
            index = ious.argmax()
            if ious[0, index] > self.iou_threshold and model_outputs["labels"][index] == label:
                score = ious[0, index] + model_outputs["scores"][index]
                output = output + score
        return output

def draw_boxes(boxes, labels, classes, confs, image, dash=False):
    labels = labels.cpu()
    for i, box in enumerate(boxes):
        color = COLORS[int(labels[i].item())]
        
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 7
        )
        cv2.putText(image, classes[i] + " " + str(np.round(confs[i].item(), 2)), (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                    lineType=cv2.LINE_AA)
    return image

def draw_dashed_bounding_box(image, bboxes, labels, thickness=6, dash_length=20):
    labels = labels.cpu()
    for i, bbox in enumerate(bboxes):
        color = COLORS[int(labels[i].item())]
        # Calculate the coordinates of the four corners of the bounding box
        x_min, y_min, x_max, y_max = bbox
        corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        
        # Draw dashed lines between consecutive corners
        for i in range(4):
            start_pt = corners[i]
            end_pt = corners[(i + 1) % 4]
            dx, dy = end_pt[0] - start_pt[0], end_pt[1] - start_pt[1]
            distance = max(abs(dx), abs(dy))
            if distance < dash_length:
                dash_length = distance - 5
            dash_count = int(distance / dash_length)
            dash_dx, dash_dy = dx / (dash_count + 0.001), dy / (dash_count+0.001)

            for j in range(dash_count):
                pt1 = int(start_pt[0] + j * dash_dx), int(start_pt[1] + j * dash_dy)
                pt2 = int(start_pt[0] + (j + 0.5) * dash_dx), int(start_pt[1] + (j + 0.5) * dash_dy)
                cv2.line(image, pt1, pt2, color, thickness)
    return image