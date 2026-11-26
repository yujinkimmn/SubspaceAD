"""Native-grid CSAM panels; display enlargement never changes the analysis grid."""
import numpy as np
from PIL import Image, ImageDraw

def jet_heatmap(values):
    import cv2
    bgr = cv2.applyColorMap((np.clip(values,0,1)*255).astype(np.uint8),cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)

def panel_metadata_lines(row, prediction=None, image_score=None):
    """Build display-only labels from the original manifest row and ground-truth mask."""
    def display(key):
        value = " ".join(str(row.get(key) or "").splitlines()).strip()
        return value or "N/A"

    label = int(row["label"])
    gt = {0: "NORMAL (0)", 1: "VOID (1)", -1: "UNLABELED (-1)"}[label]
    gt_pixels = int(np.count_nonzero(row["y"] > 0))
    lines = [f"Lot: {display('lot')}", f"Wafer: {display('wafer')}",
             f"Equipment: {display('equipment')}", f"GT: {gt} | GT pixels: {gt_pixels}"]
    if prediction is not None:
        predicted = "VOID (1)" if prediction else "NORMAL (0)"
        line = f"Image prediction: {predicted}"
        if image_score is not None:
            line += f" | Score: {float(image_score):.3g}"
        lines.append(line)
    return lines


def save_panel(x, score, threshold, path, metadata=None, *, scale=4):
    """Display native pixels as exact integer-sized blocks (no interpolation)."""
    if not isinstance(scale, int) or not 1 <= scale <= 16:
        raise ValueError("panel scale must be an integer from 1 to 16")
    h, w = x.shape
    original = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    mask = ((score > threshold) * 255).astype(np.uint8)
    panels = [Image.fromarray(original).convert("RGB"), Image.fromarray(jet_heatmap(score)),
              Image.fromarray(mask).convert("RGB")]
    gap, margin, footer = 12, 12, 64
    pw, ph = w * scale, h * scale
    width = 2 * margin + 3 * pw + 2 * gap
    lines = list(metadata or ())
    header_height = 0
    if lines:
        # Long metadata grows only the canvas, not the panel geometry.
        probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        bounds = [probe.textbbox((0, 0), text, anchor="lt") for text in lines]
        line_height = max(16, max(b[3] - b[1] for b in bounds) + 4)
        header_height = 8 + len(lines) * line_height
        width = max(width, 2 * margin + max(b[2] for b in bounds))
    top = 32 + header_height
    canvas = Image.new("RGB", (width, top + ph + footer), "white")
    draw = ImageDraw.Draw(canvas)
    for i, text in enumerate(lines):
        draw.text((margin, 6 + i * line_height), text, anchor="lt", fill="black")
    for i, (title, panel) in enumerate(zip(("Original", "Anomaly map", "Pred mask"), panels)):
        left = margin + i * (pw + gap)
        draw.text((left + pw // 2, 10 + header_height), title, anchor="mt", fill="black")
        canvas.paste(panel.resize((pw, ph), Image.Resampling.NEAREST), (left, top))
    left, y = margin + pw + gap, top + ph + 7
    bar = jet_heatmap(np.linspace(0, 1, pw, dtype=np.float64)[None, :])
    canvas.paste(Image.fromarray(bar).resize((pw, 9), Image.Resampling.NEAREST), (left, y))
    tick = left + round(np.clip(threshold, 0, 1) * (pw - 1))
    draw.line((tick, y - 2, tick, y + 12), fill="black", width=1)
    draw.text((left, y + 15), "0", fill="black")
    draw.text((left + pw, y + 15), "1", anchor="ra", fill="black")
    draw.text((margin, y + 30), f"Threshold: {threshold:.3g} | Max: {score.max():.3g} | Display: {scale}x nearest", fill="black")
    draw.text((margin, y + 44), "White = predicted defect", fill="black")
    canvas.save(path)
