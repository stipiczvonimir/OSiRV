import argparse
import pathlib
import numpy as np
import cv2
import random
import os
import matplotlib.pyplot as plt

COLORS = ["orange", "blue", "green", "cyan", "red", "yellow", "magenta", "peru", "azure", "slateblue"]

def plot_bbox(bbox_XYXY, label, color, added_labels, score):
    """Plot bounding box with confidence score for each detected digit."""
    xmin, ymin, xmax, ymax = bbox_XYXY
    plt.plot([xmin, xmin, xmax, xmax, xmin], [ymin, ymax, ymax, ymin, ymin], color=color, label=str(label) if label not in added_labels else "")
    plt.text(xmin, ymin - 5, f"{label} ({score:.2f})", color=color, fontsize=9, bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
    added_labels.add(label)


def load_templates(template_dir):
    """Load templates for digits 0-9 from the template directory."""
    templates = {str(digit): [] for digit in range(10)}

    for digit in range(10):
        digit_dir = os.path.join(template_dir, str(digit))
        if not os.path.exists(digit_dir):
            print(f"Missing digit directory: {digit_dir}")
            continue 

        for filename in os.listdir(digit_dir):
            if filename.endswith(".png"):
                template = cv2.imread(os.path.join(digit_dir, filename), cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    templates[str(digit)].append(template)
                else:
                    print(f"Failed to load template: {filename}")

    return templates

def detect_digits_with_templates(image, templates, max_threshold=0.65, scales=None):
    """Detect digits using template matching, prioritizing larger scales first."""
    if scales is None:
        scales = [2.5, 2.0, 1.5, 1.0, 0.75, 0.5]  # Largest first

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detected_boxes, detected_labels, detected_scores = [], [], []

    for scale in scales:
        for digit, template_list in templates.items():
            for template in template_list:
                resized_template = cv2.resize(template, None, fx=scale, fy=scale)

                result = cv2.matchTemplate(image_gray, resized_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                if max_val > max_threshold:
                    h, w = resized_template.shape
                    new_box = [max_loc[0], max_loc[1], max_loc[0] + w, max_loc[1] + h]

                    if not any(is_inside(new_box, existing_box) for existing_box in detected_boxes):
                        detected_boxes.append(new_box)
                        detected_labels.append(digit)
                        detected_scores.append(max_val)

    return detected_boxes, detected_labels, detected_scores




def is_inside(box1, box2):
    """Check if box1 is completely inside box2."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    return (x2_min <= x1_min and x2_max >= x1_max and y2_min <= y1_min and y2_max >= y1_max)



def non_maximum_suppression(boxes, scores, labels, iou_threshold=0.1):
    if len(boxes) == 0:
        return [], [], []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]  # Sort by confidence

    keep_boxes = []
    keep_scores = []
    keep_labels = []

    while len(order) > 0:
        i = order[0]  # Highest confidence box
        keep_boxes.append(boxes[i])
        keep_scores.append(scores[i])
        keep_labels.append(labels[i])

        to_delete = []
        for j in order[1:]:
            if is_inside(boxes[j], boxes[i]): 
                to_delete.append(j)


        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

    
        remaining_indices = np.where(iou <= iou_threshold)[0] + 1
        remaining_indices = [order[idx] for idx in remaining_indices]

        remaining_indices = [idx for idx in remaining_indices if idx not in to_delete]

        order = np.array(remaining_indices)  # Update order

    return keep_boxes, keep_scores, keep_labels



def main(directory="data/mnist_detection/test/", num_images=3):
    """Main function to process images and detect digits."""
    base_path = pathlib.Path(directory)
    image_dir = base_path.joinpath("images")

    templates = load_templates('./templates')

    impaths = list(image_dir.glob("*.png"))
    selected_impaths = random.sample(impaths, min(num_images, len(impaths)))

    for impath in selected_impaths:
        im = plt.imread(str(impath))
        im_bgr = cv2.imread(str(impath))

        detected_boxes, detected_labels, detected_scores = detect_digits_with_templates(im_bgr, templates)

        final_boxes, final_scores, final_labels = non_maximum_suppression(
            detected_boxes, detected_scores, detected_labels
        )

        plt.imshow(im, cmap="gray")
        added_labels = set()
        for bbox, digit, score in zip(final_boxes, final_labels, final_scores):
            color = COLORS[int(digit)]
            plot_bbox(bbox, digit, color, added_labels, score)

        plt.title("Detected Digits (0-9) in Image")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', facecolor='gray', framealpha=1.0)
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default="data/mnist_detection/test/", help="Directory containing the images and labels")
    parser.add_argument("--num_images", type=int, default=5, help="Number of random images to process")
    args = parser.parse_args()

    main(args.directory, args.num_images)