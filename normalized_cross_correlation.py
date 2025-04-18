import argparse
import pathlib
import numpy as np
import cv2
import random
import os
import matplotlib.pyplot as plt
from PIL import Image

COLORS = ["orange", "blue", "green", "cyan", "red", "yellow", "magenta", "peru", "azure", "slateblue"]

def plot_bbox(bbox_XYXY, label, color, added_labels, score):
    """Plot bounding box with confidence score for each detected digit."""
    xmin, ymin, xmax, ymax = bbox_XYXY
    plt.plot([xmin, xmin, xmax, xmax, xmin], [ymin, ymax, ymax, ymin, ymin], color=color, label=str(label) if label not in added_labels else "")
    plt.text(xmin, ymin - 5, f"{label} ({score:.2f})", color=color, fontsize=9, bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
    added_labels.add(label)


def load_templates(template_dir):
    """Load the first image from each shape (circle, square, star, triangle) as templates without resizing and remove background."""
    shapes = ["circle", "square", "star", "triangle"]
    templates = {}

    for shape in shapes:
        shape_dir = os.path.join(template_dir, shape)
        if os.path.exists(shape_dir):
            shape_images = [f for f in os.listdir(shape_dir) if f.endswith(".png")]
            if shape_images:
                # Select the first image from the directory
                selected_image = shape_images[0]
                image_path = os.path.join(shape_dir, selected_image)

                # Open the image and convert to RGBA (with transparency channel)
                img = Image.open(image_path).convert('RGBA')

                # Get the image data
                datas = img.getdata()
                new_data = []
                for item in datas:
                    if item[:3] == (255, 255, 255):  # If the pixel is white (255, 255, 255)
                        new_data.append((255, 255, 255, 0))  # Make it transparent
                    else:
                        new_data.append(item)

                # Update the image data with the modified pixels
                img.putdata(new_data)

                # Convert back to a numpy array (for use with OpenCV)
                template = np.array(img)

                # If the template has an alpha channel, remove it (convert to grayscale)
                if template.shape[2] == 4:
                    template = cv2.cvtColor(template, cv2.COLOR_RGBA2GRAY)

                templates[shape] = template
            else:
                print(f"No images found in {shape_dir}")
        else:
            print(f"Missing directory for shape: {shape}")

    return templates



def detect_shapes_with_templates(image, templates, threshold=0.65):
    """Detect 4 shapes using fixed-size (200x200) template matching."""
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detected_boxes, detected_labels, detected_scores = [], [], []

    for label, template in templates.items():
        result = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > threshold:
            h, w = template.shape
            new_box = [max_loc[0], max_loc[1], max_loc[0] + w, max_loc[1] + h]

            if not any(is_inside(new_box, existing_box) for existing_box in detected_boxes):
                detected_boxes.append(new_box)
                detected_labels.append(label)
                detected_scores.append(max_val)

    return detected_boxes, detected_labels, detected_scores


def is_inside(box1, box2):
    """Check if box1 is completely inside box2."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    return (x2_min <= x1_min and x2_max >= x1_max and y2_min <= y1_min and y2_max >= y1_max)


def non_maximum_suppression(boxes, scores, labels, iou_threshold=0.3):
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


def main(image_path="./output_image.jpg"):
    """Main function to process a specific image and detect shapes."""
    # Load the image and templates
    templates = load_templates('./four_shapes/2/shapes/')  # Load first images from each shape

    # Read the image to be processed
    im = plt.imread(image_path)
    im_bgr = cv2.imread(image_path)

    # Perform shape detection
    detected_boxes, detected_labels, detected_scores = detect_shapes_with_templates(im_bgr, templates)

    # Apply non-maximum suppression
    final_boxes, final_scores, final_labels = non_maximum_suppression(
        detected_boxes, detected_scores, detected_labels
    )

    # Plot results
    plt.imshow(im, cmap="gray")
    added_labels = set()
    for bbox, shape, score in zip(final_boxes, final_labels, final_scores):
        color = COLORS[hash(shape) % len(COLORS)]  # Ensure color mapping for shapes
        plot_bbox(bbox, shape, color, added_labels, score)

    plt.title("Detected Shapes (Circle, Square, Star, Triangle) in Image")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', facecolor='gray', framealpha=1.0)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", nargs="?", default="./output_image.jpg", help="Path to the specific image")
    args = parser.parse_args()

    main(args.image_path)
