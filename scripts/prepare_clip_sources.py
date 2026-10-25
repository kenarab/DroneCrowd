"""Download the pinned, hashed inference-only CLIP-EBC sources to an explicit directory."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] / "ops/clip-ebc-sources.json").read_text())
    for name, expected in manifest["files"].items():
        url = f"https://raw.githubusercontent.com/{manifest['repository']}/{manifest['commit']}/{name}"
        destination = args.output_dir / name
        if destination.exists():
            data = destination.read_bytes()
        else:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read(4 * 1024 * 1024)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Source hash mismatch: {name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Source hash mismatch: {name}")
    print(json.dumps({"commit": manifest["commit"], "verified_files": len(manifest["files"])}))


if __name__ == "__main__":
    main()
