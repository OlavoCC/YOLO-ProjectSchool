"""
Esse dataset não veio com um split real de treino/validação: as FOTOS
estavam em images/train/ e os XMLs correspondentes (mesma imagem, mesmo
nome) estavam em images/val/ -- ou seja, é UM conjunto só, separado por
engano em duas pastas durante a extração do zip.

Este script:
  1. Lê as imagens de <root>/images/train/*.png
  2. Lê os XMLs de <root>/images/val/*.xml (casando pelo nome do arquivo)
  3. Embaralha e faz um split 80/20 de verdade (nenhuma imagem se repete
     entre treino e validação)
  4. Gera uma estrutura YOLO limpa, numa pasta NOVA (pra não misturar com
     a bagunça anterior):

     <root>/data/processed/dataset_final/images/train/*.png
     <root>/data/processed/dataset_final/images/val/*.png
     <root>/data/processed/dataset_final/labels/train/*.txt
     <root>/data/processed/dataset_final/labels/val/*.txt

Uso:
    python scripts/data/convert_voc_to_yolo.py <pasta_raiz_do_dataset>
    Ex.: python scripts/data/convert_voc_to_yolo.py "C:\\Users\\temek\\OneDrive\\Desktop\\YOLO"
"""

import sys
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

CLASS_NAME_TO_ID = {"pothole": 0}
VAL_SPLIT = 0.2
SEED = 42


def convert_one_xml(xml_path: Path) -> list[str]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img_width = int(root.find("size/width").text)
    img_height = int(root.find("size/height").text)

    yolo_lines = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in CLASS_NAME_TO_ID:
            print(f"  [aviso] classe '{class_name}' não mapeada, pulando objeto em {xml_path.name}")
            continue
        class_id = CLASS_NAME_TO_ID[class_name]

        box = obj.find("bndbox")
        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        box_width = (xmax - xmin) / img_width
        box_height = (ymax - ymin) / img_height

        yolo_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )
    return yolo_lines


def find_image(stem: str, images_dir: Path) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = images_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def main():
    if len(sys.argv) < 2:
        print('Uso: python scripts/data/convert_voc_to_yolo.py "<pasta_raiz_do_dataset>"')
        sys.exit(1)

    root_dir = Path(sys.argv[1])
    images_source = root_dir / "images" / "train"   # onde estão as fotos de verdade
    xml_source = root_dir / "images" / "val"          # onde estão os xml de verdade

    if not images_source.is_dir():
        print(f"Pasta de imagens não encontrada: {images_source}")
        sys.exit(1)
    if not xml_source.is_dir():
        print(f"Pasta de xml não encontrada: {xml_source}")
        sys.exit(1)

    # Output para data/processed/dataset_final
    out_root = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "dataset_final"
    out_images_train = out_root / "images" / "train"
    out_images_val = out_root / "images" / "val"
    out_labels_train = out_root / "labels" / "train"
    out_labels_val = out_root / "labels" / "val"
    for folder in (out_images_train, out_images_val, out_labels_train, out_labels_val):
        folder.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(xml_source.glob("*.xml"))
    if not xml_files:
        print(f"Nenhum .xml encontrado em {xml_source}")
        sys.exit(1)

    # Casa cada xml com sua imagem correspondente (mesmo nome, extensão diferente)
    pairs = []
    unmatched = 0
    for xml_path in xml_files:
        image_path = find_image(xml_path.stem, images_source)
        if image_path is None:
            print(f"  [aviso] sem imagem correspondente pra {xml_path.name}, pulando")
            unmatched += 1
            continue
        pairs.append((xml_path, image_path))

    print(f"{len(pairs)} pares imagem+xml encontrados ({unmatched} sem correspondência)")

    # Embaralha e separa ANTES de processar, pra garantir que a mesma
    # imagem nunca cai nos dois lados.
    random.seed(SEED)
    random.shuffle(pairs)
    split_index = int(len(pairs) * (1 - VAL_SPLIT))
    train_pairs = pairs[:split_index]
    val_pairs = pairs[split_index:]
    print(f"Split: {len(train_pairs)} treino / {len(val_pairs)} validação")

    converted = 0
    for pair_list, img_dest, label_dest in [
        (train_pairs, out_images_train, out_labels_train),
        (val_pairs, out_images_val, out_labels_val),
    ]:
        for xml_path, image_path in pair_list:
            yolo_lines = convert_one_xml(xml_path)
            shutil.copy(image_path, img_dest / image_path.name)
            (label_dest / (xml_path.stem + ".txt")).write_text("\n".join(yolo_lines) + "\n")
            converted += 1

    print(f"\nConcluído: {converted} imagens+labels organizados em {out_root}")
    print("O config/data.yaml já aponta para essa pasta.")


if __name__ == "__main__":
    main()