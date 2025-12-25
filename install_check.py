#!/usr/bin/env python3
"""
ComfyUI-COLMAP Installation Checker

Run this script to verify all dependencies are properly installed.
"""

import sys
import subprocess


def check_python_package(package_name, import_name=None):
    """Check if a Python package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True, None
    except ImportError as e:
        return False, str(e)


def check_colmap_cli():
    """Check if COLMAP CLI is available."""
    try:
        result = subprocess.run(
            ["colmap", "help"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0, None
    except FileNotFoundError:
        return False, "COLMAP CLI not found in PATH"


def check_blender():
    """Check if Blender is available (for Alembic/FBX export)."""
    try:
        result = subprocess.run(
            ["blender", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            return True, version
        return False, "Blender found but returned error"
    except FileNotFoundError:
        return False, "Blender not found in PATH"


def main():
    print("=" * 60)
    print("ComfyUI-COLMAP Installation Checker")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Check required packages
    print("Checking required Python packages...")
    required = [
        ("numpy", "numpy"),
        ("Pillow", "PIL"),
    ]
    
    for package, import_name in required:
        ok, error = check_python_package(package, import_name)
        if ok:
            print(f"  ✓ {package}")
        else:
            print(f"  ✗ {package}: {error}")
            all_ok = False
    
    print()
    
    # Check COLMAP
    print("Checking COLMAP...")
    
    # pycolmap
    ok, error = check_python_package("pycolmap")
    if ok:
        import pycolmap
        print(f"  ✓ pycolmap {pycolmap.__version__}")
    else:
        print(f"  ⚠ pycolmap not installed (will use CLI fallback)")
    
    # COLMAP CLI
    ok, error = check_colmap_cli()
    if ok:
        print(f"  ✓ COLMAP CLI available")
    else:
        print(f"  ⚠ {error}")
        if not check_python_package("pycolmap")[0]:
            print("  ✗ ERROR: Neither pycolmap nor COLMAP CLI available!")
            all_ok = False
    
    print()
    
    # Check optional packages
    print("Checking optional packages...")
    
    # OpenCV
    ok, error = check_python_package("cv2", "cv2")
    if ok:
        import cv2
        print(f"  ✓ OpenCV {cv2.__version__} (for YAML export)")
    else:
        print(f"  ⚠ OpenCV not installed (YAML export may not work)")
    
    # Blender
    ok, version = check_blender()
    if ok:
        print(f"  ✓ {version} (for Alembic/FBX export)")
    else:
        print(f"  ⚠ {version}")
        print("    Alembic and FBX export will not be available")
    
    print()
    print("=" * 60)
    
    if all_ok:
        print("✓ All required dependencies are installed!")
        print("  ComfyUI-COLMAP is ready to use.")
    else:
        print("✗ Some required dependencies are missing.")
        print("  Please install them before using ComfyUI-COLMAP.")
        print()
        print("Quick install:")
        print("  pip install numpy Pillow pycolmap opencv-python")
    
    print("=" * 60)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
