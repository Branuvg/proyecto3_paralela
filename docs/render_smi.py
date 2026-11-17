"""Renderiza la salida real de nvidia-smi (texto guardado) como imagen. Solo omite procesos ajenos al proyecto."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
F = Path(__file__).parent / "figuras"
font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 16)
for name in ("reposo", "entrenamiento"):
    lines = (F / f"nvidia_smi_{name}.txt").read_text(encoding="utf8").splitlines()
    keep = [l for l in lines if not (l.startswith("|    0") and "python" not in l.lower())]
    w = int(max(font.getlength(l) for l in keep)) + 30
    h = 24 * len(keep) + 20
    im = Image.new("RGB", (w, h), (12, 12, 12)); d = ImageDraw.Draw(im)
    for i, l in enumerate(keep): d.text((15, 10 + 24 * i), l, font=font, fill=(204, 204, 204))
    im.save(F / f"nvidia_smi_{name}.png")
