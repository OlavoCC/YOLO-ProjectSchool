"""
Roda o modelo treinado numa imagem (ou pasta de imagens) e mostra/salva
as detecções -- útil para simular o fluxo real: cidadão tira a foto,
o backend manda pro modelo, modelo devolve se tem buraco + onde.

Uso:
    python scripts/inference/predict.py caminho/da/foto.jpg
    python scripts/inference/predict.py caminho/da/pasta/
    python scripts/inference/predict.py caminho/da/foto.jpg --model models/trained/pothole_yolov8n.pt
"""

import sys
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent.parent  # raiz do projeto (C:\YOLO)
DEFAULT_MODEL = ROOT / "models" / "trained" / "pothole_yolov8n.pt"

CONF_THRESHOLD = 0.55  # confiança mínima pra considerar detecção válida
                        # (ajuste depois de ver os resultados: subir reduz
                        # falsos positivos, descer reduz falsos negativos)


def main():
    if len(sys.argv) < 2:
        print("Uso: python scripts/inference/predict.py <imagem_ou_pasta> [--model caminho/modelo.pt]")
        sys.exit(1)

    source = sys.argv[1]
    model_path = DEFAULT_MODEL

    # Permite passar --model como argumento opcional
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            model_path = ROOT / sys.argv[idx + 1]

    if not model_path.exists():
        print(f"Modelo não encontrado em {model_path}. Rode scripts/train/train.py primeiro.")
        sys.exit(1)

    model = YOLO(str(model_path))

    results = model.predict(
        source=source,
        conf=CONF_THRESHOLD,
        save=True,           # salva as imagens com as caixas desenhadas
        project=str(ROOT / "runs"),
        name="predict",
    )

    # Para integrar com o backend (ex.: decidir se manda pro dashboard da
    # prefeitura), o que importa são as coordenadas e a confiança de cada
    # caixa detectada -- exemplo de como extrair isso em JSON:
    for r in results:
        detections = []
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            detections.append({
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
                "confidence": round(conf, 3),
            })
        print(f"\n{r.path}: {len(detections)} buraco(s) detectado(s)")
        for d in detections:
            print(f"  -> {d}")


if __name__ == "__main__":
    main()