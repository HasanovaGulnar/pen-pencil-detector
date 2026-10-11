from pathlib import Path

root = Path("dataset/roboflow")
if not root.exists():
    raise SystemExit("dataset/roboflow qovlugu tapilmadi.")

label_count = 0
for split in ("train", "valid", "test"):
    labels_dir = root / split / "labels"
    if not labels_dir.exists():
        continue

    for label_file in labels_dir.glob("*.txt"):
        lines = label_file.read_text(encoding="utf-8").splitlines()
        updated = []

        for line in lines:
            parts = line.split()
            if not parts:
                continue
            if parts[0] not in {"0", "1", "2"}:
                raise ValueError(f"Naməlum class ID: {label_file}: {parts[0]}")
            parts[0] = "0"
            updated.append(" ".join(parts))

        label_file.write_text(
            "\n".join(updated) + ("\n" if updated else ""),
            encoding="utf-8",
        )
        label_count += 1

absolute_root = (Path.cwd() / root).resolve().as_posix()
yaml_text = f"""path: {absolute_root}
train: train/images
val: valid/images
test: test/images

names:
  0: pen_pencil

# Source: Roboflow Universe / phatcara/pens-and-pencils, version 1
# Original classes: Blue-pen, Pencil, Red-pen
# License: CC BY 4.0
"""
(root / "data.yaml").write_text(yaml_text, encoding="utf-8")

print(f"Yenilənən label faylları: {label_count}")
print(f"Dataset YAML: {root / 'data.yaml'}")
