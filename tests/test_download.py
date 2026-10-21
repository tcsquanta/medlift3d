"""TCIA downloader against a local mock of the NBIA getImage endpoint (no internet needed)."""
import io
import os
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

pytest.importorskip("pydicom")

from test_dicom import _write_series  # noqa: E402

UID = "1.2.826.0.1.3680043.2.1125.7"


def _series_zip(tmp_path):
    folder = tmp_path / "src"
    _write_series(str(folder), UID, [-5.0, -2.5, 0.0, 2.5], [1, 2, 3, 4])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for k, name in enumerate(sorted(os.listdir(folder))):
            # NBIA puts files in a sub-folder; one file without the .dcm extension; plus a LICENSE
            arc = f"LIDC-IDRI-9999/{name}" if k < 3 else "LIDC-IDRI-9999/no_extension"
            zf.write(folder / name, arc)
        zf.writestr("LICENSE", "CC BY 3.0")
    return buf.getvalue()


@pytest.fixture
def nbia(tmp_path, monkeypatch):
    for k in ("http_proxy", "HTTP_PROXY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    good = _series_zip(tmp_path)
    empty = io.BytesIO()
    zipfile.ZipFile(empty, "w").close()
    state = {"requests": 0, "fail_next": True}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"] += 1
            q = parse_qs(urlparse(self.path).query)
            assert urlparse(self.path).path == "/getImage" and q.get("NewFileNames") == ["Yes"]
            if state["fail_next"]:                       # first request fails -> exercises the retry
                state["fail_next"] = False
                self.send_response(500)
                self.end_headers()
                return
            body = good if q.get("SeriesInstanceUID") == [UID] else empty.getvalue()
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", state
    server.shutdown()


def test_download_unpack_and_load(tmp_path, nbia):
    from medlift3d.data.lidc import download_series, load_dicom_series

    base, state = nbia
    root = tmp_path / "LIDC-IDRI"
    os.makedirs(root / (UID + ".part"))                  # leftover of an interrupted run
    (root / (UID + ".part") / "junk.dcm").write_bytes(b"x")
    out = download_series(UID, str(root), retries=2, base_url=base)
    assert out == str(root / UID)
    expected = sorted(os.listdir(tmp_path / "src"))[:3] + ["no_extension.dcm"]   # LICENSE skipped
    assert sorted(os.listdir(out)) == sorted(expected)
    assert not os.path.exists(str(root / UID) + ".part")
    assert state["requests"] == 2                                     # one failure + one retry
    hu, vg, z = load_dicom_series(out, UID)
    assert hu.shape == (4, 16, 20) and list(z) == [-5.0, -2.5, 0.0, 2.5]
    download_series(UID, str(root), retries=2, base_url=base)        # already there -> no request
    assert state["requests"] == 2


def test_download_wrong_uid_leaves_nothing(tmp_path, nbia):
    from medlift3d.data.lidc import download_series

    base, state = nbia
    state["fail_next"] = False
    root = tmp_path / "LIDC-IDRI"
    with pytest.raises(RuntimeError, match="no DICOM"):
        download_series("9.9.9", str(root), retries=1, base_url=base)
    assert not os.path.exists(root / "9.9.9") and not os.path.exists(str(root / "9.9.9") + ".part")
