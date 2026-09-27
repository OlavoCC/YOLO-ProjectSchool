"""
Treino de um YOLOv8 (transfer learning) para detectar buracos no asfalto.

Uso:
    python scripts/train/train.py

Antes de rodar:
    1. Dataset deve estar em data/processed/dataset_final/ (criado pelo scripts/data/convert_voc_to_yolo.py)
    2. Confira config/data.yaml (nc e names).
    3. pip install -r requirements.txt
"""

from pathlib import Path
from ultralytics import YOLO

# --- Config ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent  # raiz do projeto (C:\YOLO)
DATA_YAML = ROOT / "config" / "data.yaml"
BASE_MODEL = ROOT / "models" / "base" / "yolov8n.pt"

EPOCHS = 100
IMG_SIZE = 640
BATCH = 16
PATIENCE = 20
RUN_NAME = "buraco_v1"


def main():
    # Carrega os pesos pré-treinados no COCO. É aqui que entra o transfer
    # learning: a rede já sabe extrair bordas/texturas/formas; vamos só
    # reajustar (fine-tune) tudo para reconhecer a classe "pothole".
    model = YOLO(str(BASE_MODEL))

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        patience=PATIENCE,
        name=RUN_NAME,
        project=str(ROOT / "runs"),

        # Augmentation: como o dataset provavelmente é pequeno, essas
        # transformações ajudam o modelo a não decorar as fotos e sim
        # aprender a "forma" de um buraco em condições variadas de luz,
        # ângulo e distância.
        hsv_h=0.015,   # variação leve de matiz (diferentes tipos de asfalto/luz)
        hsv_s=0.7,     # variação de saturação
        hsv_v=0.4,     # variação de brilho (sol/sombra)
        degrees=10,    # rotação leve (celular não fica sempre reto)
        translate=0.1,
        scale=0.5,     # simula buracos fotografados de perto/longe
        flipud=0.0,    # buraco de cabeça para baixo não faz sentido
        fliplr=0.5,    # espelhar horizontalmente é válido
        mosaic=1.0,    # combina 4 imagens em 1 -- muito eficaz com poucos dados
    )

    # Roda validação final e imprime métricas (mAP50, mAP50-95, precision, recall)
    metrics = model.val()
    print(metrics)

    best_path = ROOT / "runs" / RUN_NAME / "weights" / "best.pt"
    print(f"\nMelhor checkpoint salvo em: {best_path}")
    print("Use esse arquivo em scripts/inference/predict.py")


if __name__ == "__main__":
    main()