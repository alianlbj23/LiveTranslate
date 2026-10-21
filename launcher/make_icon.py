"""Writes assets/LiveTranslate.ico (the icon of LiveTranslate.exe) with the same drawing as
main.create_app_icon(), at the sizes Windows uses. Run again after changing that icon:

    .venv/Scripts/python.exe -I launcher/make_icon.py"""
import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZES = (16, 24, 32, 48, 64, 128, 256)


def _png(size: int) -> bytes:
    from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, QRectF, Qt
    from PyQt6.QtGui import QColor, QFont, QImage, QPainter

    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 0))
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    s = size / 64.0  # main.create_app_icon draws on 64 x 64
    p.setBrush(QColor(60, 130, 240))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(QRectF(4 * s, 4 * s, 56 * s, 56 * s), 12 * s, 12 * s)
    p.setPen(QColor(255, 255, 255))
    font = QFont("Consolas")
    font.setBold(True)
    font.setPixelSize(max(6, round(37 * s)))  # 28 pt at 96 dpi
    p.setFont(font)
    p.drawText(QRectF(0, 0, size, size), Qt.AlignmentFlag.AlignCenter, "LT")
    p.end()
    data = QByteArray()
    buf = QBuffer(data)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    return bytes(data)


def build_ico(sizes=SIZES) -> bytes:
    """ICO with one PNG-compressed image per size (supported since Windows Vista)."""
    images = [(size, _png(size)) for size in sizes]
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = len(header) + 16 * len(images)
    entries, blobs = b"", b""
    for size, png in images:
        dim = 0 if size >= 256 else size  # 0 means 256
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(png), offset)
        blobs += png
        offset += len(png)
    return header + entries + blobs


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)  # fonts need an application
    out = os.path.join(ROOT, "assets", "LiveTranslate.ico")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(build_ico())
    print(out)
    del app
    return 0


if __name__ == "__main__":
    sys.exit(main())
