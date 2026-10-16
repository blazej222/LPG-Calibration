"""Install an official prebuilt rootable Android image; no Android source build."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile



def download(url: str, path: Path, size: int) -> None:
    if path.exists() and path.stat().st_size == size:
        return
    last_error = None
    for attempt in range(1, 7):
        offset = path.stat().st_size if path.exists() else 0
        if offset > size:
            raise RuntimeError(f"Unexpected oversized download: {path}")
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=30) as source:
                if offset and source.status != 206:
                    offset = 0
                last_report = time.monotonic()
                with path.open("ab" if offset else "wb") as output:
                    while block := source.read(256 * 1024):
                        output.write(block)
                        offset += len(block)
                        if time.monotonic() - last_report > 10:
                            output.flush()
                            print(f"Downloaded {offset / 1048576:.1f}/{size / 1048576:.1f} MiB", flush=True)
                            last_report = time.monotonic()
            if offset == size:
                return
            last_error = RuntimeError(f"Short download: {offset}/{size}")
        except (OSError, TimeoutError) as error:
            last_error = error
        print(f"Download retry {attempt}/6: {last_error}", flush=True)
    raise RuntimeError(f"Download did not complete: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--api", type=int, default=36)
    parser.add_argument("--tag", choices=("default", "google_apis"), default="google_apis")
    args = parser.parse_args()
    repository = "https://dl.google.com/android/repository/sys-img/" + ("android" if args.tag == "default" else args.tag) + "/"
    package_path = f"system-images;android-{args.api};{args.tag};x86_64"
    dashboard = Path(__file__).resolve().parents[1]
    cache = dashboard / "build/downloads"
    cache.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(repository + "sys-img2-3.xml", timeout=30) as response:
        metadata = response.read()
    (cache / f"{args.tag}-repository.xml").write_bytes(metadata)
    root = ET.fromstring(metadata)
    package = next(p for p in root.findall("remotePackage") if p.attrib["path"] == package_path)
    archive = next(a for a in package.findall("archives/archive") if a.findtext("host-os") in ("windows", None))
    filename = archive.findtext("complete/url")
    expected_size = int(archive.findtext("complete/size"))
    checksum = archive.find("complete/checksum")
    expected_hash = checksum.text
    algorithm = checksum.attrib["type"]
    path = cache / filename
    print(f"Package: {package_path}; {expected_size} bytes; {repository + filename}", flush=True)
    download(repository + filename, path, expected_size)
    with path.open("rb") as downloaded:
        actual_hash = hashlib.file_digest(downloaded, algorithm).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(f"Archive checksum mismatch: {actual_hash} != {expected_hash}")
    print(f"Verified {algorithm}: {actual_hash}", flush=True)
    destination = args.sdk_root.resolve() / f"system-images/android-{args.api}/{args.tag}"
    image = destination / "x86_64"
    if image.exists():
        if not (image / "source.properties").is_file():
            raise RuntimeError(f"Refusing to overwrite an existing incomplete image: {image}")
        properties = dict(line.split("=", 1) for line in (image / "source.properties").read_text().splitlines()
                          if "=" in line and not line.startswith("#"))
        if properties.get("AndroidVersion.ApiLevel") != str(args.api) or properties.get("Pkg.Revision") != package.findtext("revision/major"):
            raise RuntimeError(f"An existing SDK image has another version; refusing to overwrite: {image}")
        print(f"Image directory already exists: {image}", flush=True)
    else:
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as zipped:
            for entry in zipped.infolist():
                target = (destination / entry.filename).resolve()
                if not target.is_relative_to(destination.resolve()) or not entry.filename.startswith("x86_64/"):
                    raise RuntimeError(f"Unexpected archive path: {entry.filename}")
            zipped.extractall(destination)
        print(f"Extracted official image to: {image}", flush=True)
    # Recent images initialize userdata from data/ rather than shipping userdata.img.
    for required in ("system.img", "ramdisk.img", "kernel-ranchu", "source.properties"):
        if not (image / required).is_file():
            raise RuntimeError(f"Required image file is missing: {required}")
        with zipfile.ZipFile(path) as zipped:
            if (image / required).stat().st_size != zipped.getinfo(f"x86_64/{required}").file_size:
                raise RuntimeError(f"Incomplete or modified SDK image file: {required}")
    record = {"package": package_path, "url": repository + filename, "size": expected_size,
              "checksum_algorithm": algorithm, "checksum": actual_hash, "image_directory": str(image)}
    (cache / f"android-{args.api}-{args.tag}-install-record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Android API {args.api} {args.tag} image ready", flush=True)


if __name__ == "__main__":
    main()
