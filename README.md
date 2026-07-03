# EfficientNetV2L — OCR российских автомобильных номеров (ONNX)

Предобученная модель оптического распознавания символов (OCR) на номерных знаках российских автомобилей.  
Экспортирована в формат **ONNX** для использования с [ONNX Runtime](https://onnxruntime.ai/) и совместимыми бэкендами (CPU, CUDA, DirectML, OpenVINO, TensorRT и др.).

> Модель является частью пайплайна **AY000554**: детекция номера ([YOLOv9c](https://github.com/AY000554/Car_plate_detecting)) + распознавание символов (EfficientNetV2L).

---

## Скачать модель

### Из репозитория (Git LFS)

```bash
git lfs install
git clone https://github.com/1os/EfficientNetV2L_AY000554.git
```

### Из Releases (прямая ссылка)

**[EfficientNetV2L.onnx](https://github.com/1os/EfficientNetV2L_AY000554/releases/download/v1.0.0/EfficientNetV2L.onnx)** — v1.0.0

```bash
curl -L -o EfficientNetV2L.onnx \
  https://github.com/1os/EfficientNetV2L_AY000554/releases/download/v1.0.0/EfficientNetV2L.onnx
```

---

## Кратко

| Параметр | Значение |
|----------|----------|
| Архитектура | [EfficientNetV2L](https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet_v2/EfficientNetV2L) |
| Задача | OCR символов ГРЗ (российские номера) |
| Формат | ONNX (IR v7, opset 13) |
| Конвертер | tf2onnx 1.16.1 |
| Размер файла | ~127 МБ (`EfficientNetV2L.onnx`) |
| Входное изображение | 200 × 50 px, RGB |
| Точность (номер целиком) | **94%** |
| Датасет обучения | [nomeroff.net](https://nomeroff.net.ua/) |

---

## Метрики

Результаты на тестовой выборке:

| Метрика | Значение |
|---------|----------|
| Доля распознанных **номеров** целиком | **94%** |
| Доля распознанных **символов** | ~98% |

> Для полного ANPR-пайплайна (детекция + OCR) итоговая точность зависит также от качества детектора и условий съёмки.

---

## Структура репозитория

```
.
├── EfficientNetV2L.onnx   # веса модели (Git LFS)
├── .gitattributes
├── README.md
└── LICENSE
```

> Файл `EfficientNetV2L.onnx` (~127 МБ) хранится в репозитории через **Git LFS**.  
> Для клонирования: `git lfs install && git clone ...`  
> Альтернативно — скачать напрямую из [Releases](https://github.com/1os/EfficientNetV2L_AY000554/releases/latest).

---

## Спецификация модели

### Вход

| Имя | Форма | Тип | Описание |
|-----|-------|-----|----------|
| `image` | `[N, 200, 50, 3]` | `float32` | Вырезанный и нормализованный фрагмент номера, RGB, порядок **NHWC** |

### Выход

| Имя | Форма | Тип | Описание |
|-----|-------|-----|----------|
| `dense1` | `[N, T, 25]` | `float32` | Логиты по 25 классам символов для каждой позиции последовательности |

### Алфавит (25 классов)

```
0 1 2 3 4 5 6 7 8 9
A B E K M H O P C T Y X
-
```

- Буквы — **латиница**, соответствующая российским буквам ГОСТ на номерах.
- Для номеров с **двузначным регионом** вместо третьей цифры региона используется символ `-`  
  (например: `A123BC77` → `A123BC7-`).

---

## Быстрый старт

### Установка зависимостей

```bash
pip install onnxruntime numpy opencv-python
```

Скачайте модель из [Releases](https://github.com/1os/EfficientNetV2L_AY000554/releases/latest) и положите `EfficientNetV2L.onnx` в рабочую папку.

### Пример инференса (Python)

```python
import cv2
import numpy as np
import onnxruntime as ort

ALPHABET = "0123456789ABEKMHOPCTYX-"
MODEL_PATH = "EfficientNetV2L.onnx"

def preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    """Подготовка вырезанного номера к подаче в модель."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (50, 200))  # width=50, height=200
    x = resized.astype(np.float32) / 255.0
    return np.expand_dims(x, axis=0)  # [1, 200, 50, 3]

def decode(logits: np.ndarray) -> str:
    """CTC-подобное декодирование: argmax по каждой позиции, схлопывание повторов."""
    indices = np.argmax(logits, axis=-1)[0]
    chars = []
    prev = -1
    for idx in indices:
        if idx != prev:
            chars.append(ALPHABET[idx])
        prev = idx
    return "".join(chars).strip("-")

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
plate_crop = cv2.imread("plate_crop.jpg")  # вырез номера после детектора
input_tensor = preprocess(plate_crop)
logits = session.run(None, {"image": input_tensor})[0]
plate_text = decode(logits)
print(plate_text)
```

> Нормализация и постобработка могут отличаться в зависимости от пайплайна обучения.  
> При интеграции сверяйте предобработку с [исходным проектом](https://github.com/AY000554/ocr_car_plate).

---

## Рекомендуемый пайплайн ANPR

```
Кадр с камеры
    │
    ▼
YOLOv9c — детекция номера на кадре
    │
    ▼
Вырез и выравнивание области номера (200×50)
    │
    ▼
EfficientNetV2L (этот репозиторий) — OCR символов
    │
    ▼
Строка номера (например, A123BC77)
```

Связанные репозитории:

- [AY000554/Car_plate_detecting](https://github.com/AY000554/Car_plate_detecting) — детекция номера (YOLOv9c)
- [AY000554/ocr_car_plate](https://github.com/AY000554/ocr_car_plate) — обучение и исходный код OCR-модели
- [AY000554/Car_plate_detecting_dataset](https://huggingface.co/datasets/AY000554/Car_plate_detecting_dataset) — датасет для детекции

---

## Параметры обучения

| Параметр | Значение |
|----------|----------|
| Фреймворк | TensorFlow 2.14 |
| Эпохи | 200 |
| Batch size | 32 |
| Размер входа | 200 × 50 px |
| Learning rate | 1e-4 → 0 (CosineDecay) |
| Warmup | 5 эпох |
| Разбиение данных | train 80% / val 10% / test 10% |

---

## Системные требования

- **ONNX Runtime** ≥ 1.14 (или совместимый рантайм)
- RAM: ≥ 512 МБ на одну сессию инференса
- Поддерживаемые провайдеры: CPU, CUDA, DirectML, OpenVINO, TensorRT

---

## Ограничения

- Модель обучена на **российских номерах** одного типа (стандартный ГРЗ).
- Качество падает при сильных искажениях, засветах, грязи, нестандартных ракурсах.
- Для работы в продакшене рекомендуется использовать вместе с детектором YOLOv9c из пайплайна AY000554.
- Модель **не выполняет детекцию** — только распознавание уже вырезанного фрагмента номера.

---

## Лицензия

Уточните лицензию перед публикацией. Исходный проект AY000554 и датасет nomeroff.net могут накладывать дополнительные условия использования.

---

## Ссылки

- [AY000554/ocr_car_plate](https://github.com/AY000554/ocr_car_plate)
- [AY000554/Car_plate_detecting](https://github.com/AY000554/Car_plate_detecting)
- [Nomeroff Net](https://nomeroff.net.ua/)
- [ONNX Runtime](https://onnxruntime.ai/)

---

## Цитирование

Если вы используете эту модель в своём проекте, пожалуйста, укажите ссылку на репозиторий AY000554:

```bibtex
@misc{ay000554_ocr_efficientnetv2l,
  author       = {AY000554},
  title        = {EfficientNetV2L OCR for Russian License Plates (ONNX)},
  year         = {2026},
  howpublished = {\url{https://github.com/AY000554/ocr_car_plate}}
}
```
