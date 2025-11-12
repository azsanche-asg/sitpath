import datetime
import hashlib
import json
import os


def file_hash(path, block_size=65536):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def update_manifest(manifest_path="artifact_manifest.json"):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    now = datetime.datetime.now().isoformat()
    manifest["timestamp"] = now

    for art in manifest["artifacts"]:
        path = art["path"]
        if os.path.isfile(path):
            art["hash"] = file_hash(path)
        elif os.path.isdir(path):
            art["hash"] = "<dir>"
        else:
            art["hash"] = "<missing>"

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"✅ Manifest updated at {now}")


if __name__ == "__main__":
    update_manifest()
