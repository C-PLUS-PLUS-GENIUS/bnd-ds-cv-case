import cv2
import numpy as np
import argparse
import onnxruntime as ort
from tqdm import tqdm
from torchvision import transforms


def people_detection(model_name: str, input_path: str, output_path: str):

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']

    print(f'Загрузка модели {model_name}...')
    if model_name == 'yolov5':
        onnx_model_path = 'weights/yolov5s.onnx'
        process = process_yolo
    elif model_name == 'ssd':
        onnx_model_path = 'weights/ssd300_vgg16.onnx'
        process = process_ssd

    ort_session = ort.InferenceSession(onnx_model_path, providers=providers)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Ошибка при открытии видео: {input_path}")
        raise SystemExit

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Запуск инференса с использованием модели {model_name}")
    for _ in tqdm(range(total_frames), desc="Обработка видео"):
        ret, frame = cap.read()
        if not ret:
            break
        process(ort_session, frame, width, height)
        out.write(frame)

    cap.release()
    out.release()
    print(f"Инференс завершен. Результат сохранен в {output_path}")


def non_max_suppression(detections, conf_thresh=0.2, iou_thresh=0.5):
    boxes = []
    confidences = []
    class_ids = []
    
    for det in detections:
        x_center, y_center, box_width, box_height = det[0:4]
        conf = det[4] 
        class_probs = det[5:]
        
        if conf >= conf_thresh:
            x1 = x_center - box_width / 2
            y1 = y_center - box_height / 2
            x2 = x_center + box_width / 2
            y2 = y_center + box_height / 2

            cls = np.argmax(class_probs)
            boxes.append([x1, y1, x2, y2])
            confidences.append(float(conf))
            class_ids.append(cls)

    boxes = np.array(boxes)
    confidences = np.array(confidences)
    class_ids = np.array(class_ids)

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), conf_thresh, iou_thresh)

    filtered_boxes = []
    filtered_confidences = []
    filtered_class_ids = []
    
    if len(indices) > 0:
        for i in indices.flatten():
            filtered_boxes.append(boxes[i])
            filtered_confidences.append(confidences[i])
            filtered_class_ids.append(class_ids[i])
    
    return filtered_boxes, filtered_confidences, filtered_class_ids
def process_yolo(ort_session, frame, width, height):
    resized = cv2.resize(frame, (640, 640))
    resized = resized.astype(np.float32) / 255.0
    resized = np.transpose(resized, (2, 0, 1))  # HWC to CHW
    input_tensor = np.expand_dims(resized, axis=0)

    inputs = {ort_session.get_inputs()[0].name: input_tensor}
    outputs = ort_session.run(None, inputs)

    detections = outputs[0] 
    detections = detections[0] 

    filtered_boxes, filtered_confidences, filtered_class_ids = non_max_suppression(detections)

    x_scale = width / 640
    y_scale = height / 640

    for box, conf, cls in zip(filtered_boxes, filtered_confidences, filtered_class_ids):
        if cls == 0:  # "person"
            x1, y1, x2, y2 = box
            x1 *= x_scale
            y1 *= y_scale
            x2 *= x_scale
            y2 *= y_scale

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{conf:.2f}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

def process_ssd(ort_session, frame, width, height):

    ssd_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    frame_transformed = ssd_transform(frame).unsqueeze(0).numpy()

    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: frame_transformed})

    boxes, scores, labels = outputs[0], outputs[1], outputs[2]

    for i in range(boxes.shape[0]):
        score = scores[i]
        label = labels[i]

        if score > 0.5 and label == 1:
            x1, y1, x2, y2 = boxes[i]
            x1 = int(x1 / 300 * width)
            y1 = int(y1 / 300 * height)
            x2 = int(x2 / 300 * width)
            y2 = int(y2 / 300 * height)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{score:.2f}', (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)



def main():
    parser = argparse.ArgumentParser(description='Выберите модель для инференса (yolov5 или ssd) и укажите пути к входному и выходному файлам.')
    parser.add_argument('--model-name', type=str, choices=['yolov5', 'ssd'], help='Модель для детекции: yolov5, ssd')
    parser.add_argument('--input', type=str, default='data/crawd.mp4', help='Путь к входному видеофайлу')
    parser.add_argument('--output', type=str, help='Путь к выходному видеофайлу')
    params = parser.parse_args()

    people_detection(model_name = params.model_name, input_path = params.input, output_path = params.output)

if __name__ == "__main__":
    main()