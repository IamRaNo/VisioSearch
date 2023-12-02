import onnxruntime as ort
from flask import request, Flask, jsonify
from PIL import Image
import numpy as np

app = Flask(__name__, static_url_path='/static')

# Site main page handler function.


@app.route("/")
def root():

    with open("index.html") as file:
        return file.read()

 # Handler of /detect POST endpoint
 # Receives uploaded file with a name "image_file", passes it
 # through YOLOv8 object detection network and returns and array
 # of bounding boxes.


@app.route("/detect", methods=["POST"])
def detect():

    buf = request.files["image_file"]
    boxes = detect_objects_on_image(buf.stream)
    return jsonify(boxes)

# Function receives an image, passes it through YOLOv8 neural network and returns an array of detected objects and their bounding boxes


def detect_objects_on_image(buf):

    input, img_width, img_height = prepare_input(buf)
    output = run_model(input)
    return process_output(output, img_width, img_height)

# Function used to convert input image to tensor,required as an input to YOLOv8 object detection  network.


def prepare_input(buf):

    img = Image.open(buf)
    img_width, img_height = img.size
    img = img.resize((640, 640))
    img = img.convert("RGB")
    input = np.array(img) / 255.0
    input = input.transpose(2, 0, 1)
    input = input.reshape(1, 3, 640, 640)
    return input.astype(np.float32), img_width, img_height

# Function used to pass provided input tensor to
# YOLOv8 neural network and return result


def run_model(input):

    model = ort.InferenceSession("best.onnx", providers=[
                                 'CPUExecutionProvider'])
    outputs = model.run(["output0"], {"images": input})
    return outputs[0]

# Function used to convert RAW output from YOLOv8 to an array
# of detected objects. Each object contain the bounding box of
# this object, the type of object and the probabilit


def process_output(output, img_width, img_height):

    output = output[0].astype(float)
    output = output.transpose()

    boxes = []
    for row in output:
        prob = row[4:].max()
        if prob < 0.5:
            continue
        class_id = row[4:].argmax()
        label = classes[class_id]
        xc, yc, w, h = row[:4]
        x1 = (xc - w/2) / 640 * img_width
        y1 = (yc - h/2) / 640 * img_height
        x2 = (xc + w/2) / 640 * img_width
        y2 = (yc + h/2) / 640 * img_height
        boxes.append([x1, y1, x2, y2, label, prob])

    boxes.sort(key=lambda x: x[5], reverse=True)
    result = []
    while len(boxes) > 0:
        result.append(boxes[0])
        boxes = [box for box in boxes if iou(box, boxes[0]) < 0.5]

    return result

# Function calculates "Intersection-over-union" coefficient for specified two boxes


def iou(box1, box2):

    return intersection(box1, box2)/union(box1, box2)

# Function calculates union area of two boxes


def union(box1, box2):

    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
    box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
    return box1_area + box2_area - intersection(box1, box2)

# Function calculates intersection area of two boxes


def intersection(box1, box2):

    box1_x1, box1_y1, box1_x2, box1_y2 = box1[:4]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[:4]
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)
    return (x2-x1)*(y2-y1)


# Array of YOLOv8 class labels
classes = [
    "human", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

if __name__ == "__main__":
    app.run(debug=True)
