# bnd-ds-cv-case

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/C-PLUS-PLUS-GENIUS/bnd-ds-cv-case)

## Тестовое задание на вакансию DS в BND

#### Описание
Реализация инференса двух моделей - YOLOv5s и SSD300_vgg16 для задачи детекции объектов на видео. 

Цель — сравнить производительность и качество распознавания людей в видеопотоке.

#### Клонирование репозитория

```bash
git clone https://github.com/C-PLUS-PLUS-GENIUS/bnd-ds-cv-case.git
cd bnd-ds-cv-case
```

#### Установка необходимых библиотек

```bash
pip install -r requirements.txt
```

## Использование GPU

Для запуска в режиме Cuda на устройстве должны быть установлены NVIDIA CUDA + CuDNN

Require cuDNN 9.* and CUDA 12.*, and the latest MSVC runtime.

#### Веса моделей

**Note:** Веса в FP32.

| Model Name   | ONNX Model                                                                                                     | Number of Parameters | Model Size |
| ------------ | -------------------------------------------------------------------------------------------------------------- | -------------------- | ---------- |
| YOLOv5s      | [yolov5s.onnx](https://github.com/C-PLUS-PLUS-GENIUS/bnd-ds-cv-case/tree/main/weights/yolov5s.onnx)            | 7.2M                 | 28 MB      |
| SSD300_vgg16 | [ssd300_vgg16.onnx](https://github.com/C-PLUS-PLUS-GENIUS/bnd-ds-cv-case/tree/main/weights/ssd300_vgg16.onnx)  | 35.6M                | 140 MB     |

#### Inference

Файл research.ipynb с загрузкой предобученных моделей из hub (требуется подключение к интернету)

Либо реализация через ONNX-runtime


```bash
python main.py --model-name yolov5 --input data/crowd.mp4 --output data/crowd_yolov5.mp4

python main.py --model-name ssd --input data/crowd.mp4 --output data/crowd_ssd.mp4
```

- 1-й аргумент - название модели: yolov5, ssd
- 2-й аргумент - входное видео
- 3-й аргумент - выходное видео

#### Результат для YOLOv5s
<video controls autoplay loop src="https://github.com/user-attachments/assets/17eff7ee-439d-46bd-8be3-9b1b26699c22" muted="false" width="100%"></video>
Link: https://youtu.be/uNP3gTlHcKk

#### Результат для SSD300_vgg16
<video controls autoplay loop src="https://github.com/user-attachments/assets/1bf52331-f117-4381-8653-89e99cd76101" muted="false" width="100%"></video>
Link: https://youtu.be/KznGhV021Dg

#### Предложения по улучшению результата детектирования для YOLOv5
1. Использование более мощной версии YOLOv5
2. Fine-Tuning 

#### Предложения по улучшению результата детектирования для SSD300_VGG16
1. Применение модели с увеличенным размером входных данных
2. Применение более мощной модели с ResNet, EfficientNet или MobileNet вместо VGG16
2. Fine-Tuning 
