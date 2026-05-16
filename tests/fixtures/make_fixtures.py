"""
Regenerate test fixtures.  Run once from the repo root:
    python tests/fixtures/make_fixtures.py

Produces:
    tests/fixtures/dataset_sample.zip
        forget/image_forget.png  (1x1 red PNG)
        retain/image_retain.png  (1x1 green PNG)
"""
import io
import os
import struct
import zipfile
import zlib

_HERE = os.path.dirname(os.path.abspath(__file__))


def _make_png(color: tuple) -> bytes:
    """Return the bytes of a minimal 1x1 PNG with the given RGB colour."""
    width, height = 1, 1

    def chunk(tag: bytes, data: bytes) -> bytes:
        payload = tag + data
        return struct.pack('>I', len(data)) + payload + struct.pack('>I', zlib.crc32(payload) & 0xffffffff)

    signature = b'\x89PNG\r\n\x1a\n'
    ihdr = chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
    raw = b'\x00' + bytes(color)
    idat = chunk(b'IDAT', zlib.compress(raw))
    iend = chunk(b'IEND', b'')
    return signature + ihdr + idat + iend


def make_dataset_sample() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr('forget/image_forget.png', _make_png((255, 0, 0)))
        zf.writestr('retain/image_retain.png', _make_png((0, 255, 0)))

    out = os.path.join(_HERE, 'dataset_sample.zip')
    with open(out, 'wb') as f:
        f.write(buf.getvalue())
    print(f"Written {out} ({len(buf.getvalue())} bytes)")


if __name__ == '__main__':
    make_dataset_sample()
