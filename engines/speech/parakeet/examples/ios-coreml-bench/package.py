#!/usr/bin/env python3
"""Package the physical-iPhone benchmark with user-supplied signing assets."""
import argparse
import plistlib
import shutil
import subprocess
from pathlib import Path

EXECUTABLE_NAME = "QVACSpeechBench"
MINIMUM_IOS_VERSION = "18.0"


def copy_models(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for entry in source.iterdir():
        if entry.suffix == ".mlmodelc" and entry.is_dir():
            shutil.copytree(entry, destination / entry.name, dirs_exist_ok=True)
        elif entry.suffix == ".gguf" and entry.is_file():
            shutil.copyfile(entry, destination / entry.name)


def write_signing_files(args):
    profile = plistlib.loads(subprocess.check_output(
        ["security", "cms", "-D", "-i", str(args.profile)]))
    identifier = profile["Entitlements"]["application-identifier"]
    if not identifier.endswith("." + args.bundle_id):
        raise ValueError("--bundle-id must match the supplied provisioning profile")
    shutil.copyfile(args.profile, args.output / "embedded.mobileprovision")
    entitlements = args.output.parent / (args.output.stem + "-entitlements.plist")
    entitlements.write_bytes(plistlib.dumps(profile["Entitlements"]))
    return entitlements


def package(args):
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.binary, args.output / EXECUTABLE_NAME)
    info = {
        "CFBundleIdentifier": args.bundle_id,
        "CFBundleExecutable": EXECUTABLE_NAME,
        "CFBundleName": "TranscribePsy EU Bench",
        "CFBundlePackageType": "APPL",
        "CFBundleVersion": "1",
        "CFBundleShortVersionString": "1.0",
        "LSRequiresIPhoneOS": True,
        "MinimumOSVersion": MINIMUM_IOS_VERSION,
        "UIDeviceFamily": [1],
        "UILaunchScreen": {},
    }
    (args.output / "Info.plist").write_bytes(plistlib.dumps(info))
    for value in args.model_dir:
        name, separator, directory = value.partition("=")
        if not separator or not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("--model-dir requires a simple NAME=PATH")
        copy_models(Path(directory), args.output / name)
    for wav in args.samples.glob("*.wav"):
        shutil.copyfile(wav, args.output / wav.name)
    entitlements = write_signing_files(args)
    subprocess.run(["codesign", "--force", "--sign", args.identity,
                    "--entitlements", str(entitlements), str(args.output)], check=True)
    print(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--model-dir", action="append", required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    package(parser.parse_args())


if __name__ == "__main__":
    main()
