# DLandML — ASVspoof2019 LA с LightCNN

Код для детекции спуфинга речи на датасете **ASVspoof2019 LA**.  

---

## Установка

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Быстрый старт

### 1) Тренировка

```bash
# пример: 12 эпох, батч 64
python train.py   trainer.epochs=12
```

### 2) Оценка стандартная (один кроп)

```bash
python inference.py +inferencer.from_pretrained=checkpoints/<run>/model_best.pth
```

---

## CometML

Установите ключ и (при желании) воркспейс:

```bash
export COMET_API_KEY=...        
```

Проект/режим задаются в конфиге (`src/configs/asvspoof.yaml`).

---
