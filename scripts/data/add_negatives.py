"""
Adiciona imagens NEGATIVAS (asfalto liso, faixas, etc) ao dataset_final/
Labels vazios = "esta imagem nao contem buracos"

Dataset fonte: <root>/data/external/Dataset (ou caminho configurável)
Destino: <root>/data/processed/dataset_final/
"""

import shutil
from pathlib import Path

# Caminhos configuráveis
SRC = Path(__file__).resolve().parent.parent.parent / "data" / "external" / "Dataset"
DST = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "dataset_final"


def copy_negatives(split_name):
    src_img = SRC / split_name / "Normal"
    dst_img = DST / "images" / split_name
    dst_lbl = DST / "labels" / split_name

    if not src_img.exists():
        print(f"  [aviso] {src_img} nao existe, pulando")
        return 0

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    count = 0
    for img in src_img.glob("*.*"):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            continue
        dst_file = dst_img / img.name
        if dst_file.exists():
            base = img.stem
            ext = img.suffix
            counter = 1
            while dst_file.exists():
                dst_file = dst_img / f"{base}_neg{counter}{ext}"
                counter += 1

        shutil.copy2(img, dst_file)
        (dst_lbl / (dst_file.stem + ".txt")).write_text("")
        count += 1

    print(f"  {split_name}/Normal: {count} negativos adicionados")
    return count


if __name__ == "__main__":
    total = 0
    for split in ["train", "val"]:
        total += copy_negatives(split)

    print(f"\nTotal: {total} negativos adicionados ao data/processed/dataset_final/")
    print("Agora rode: python scripts/train/train.py")