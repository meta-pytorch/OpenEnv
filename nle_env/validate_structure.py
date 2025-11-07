#!/usr/bin/env python3
"""
Code structure validation for NLE environment.

This script validates that all files exist and have the correct structure
without requiring dependencies to be installed.
"""

import ast
from pathlib import Path
import sys

print("=" * 70)
print("NLE Environment Code Structure Validation")
print("=" * 70)

base_path = Path(__file__).parent

# Test 1: Check all files exist
print("\n[1/6] Checking file structure...")
required_files = [
    "__init__.py",
    "models.py",
    "client.py",
    "README.md",
    "server/__init__.py",
    "server/app.py",
    "server/nle_environment.py",
    "server/Dockerfile",
]

missing_files = []
for file_path in required_files:
    full_path = base_path / file_path
    if not full_path.exists():
        missing_files.append(file_path)
        print(f"  ✗ Missing: {file_path}")
    else:
        print(f"  ✓ Found: {file_path}")

if missing_files:
    print(f"\n✗ Missing {len(missing_files)} files")
    sys.exit(1)

# Test 2: Validate models.py
print("\n[2/6] Validating models.py...")
models_path = base_path / "models.py"
with open(models_path) as f:
    try:
        tree = ast.parse(f.read())
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        required_classes = ["NLEAction", "NLEObservation", "NLEState"]
        for cls in required_classes:
            if cls in classes:
                print(f"  ✓ Found class: {cls}")
            else:
                print(f"  ✗ Missing class: {cls}")
                sys.exit(1)
    except SyntaxError as e:
        print(f"  ✗ Syntax error in models.py: {e}")
        sys.exit(1)

# Test 3: Validate client.py
print("\n[3/6] Validating client.py...")
client_path = base_path / "client.py"
with open(client_path) as f:
    try:
        tree = ast.parse(f.read())
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        if "NLEEnv" in classes:
            print(f"  ✓ Found class: NLEEnv")
        else:
            print(f"  ✗ Missing class: NLEEnv")
            sys.exit(1)

        # Check for required methods
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "NLEEnv":
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                required_methods = ["_step_payload", "_parse_result", "_parse_state"]
                for method in required_methods:
                    if method in methods:
                        print(f"  ✓ Found method: {method}")
                    else:
                        print(f"  ✗ Missing method: {method}")
                        sys.exit(1)
    except SyntaxError as e:
        print(f"  ✗ Syntax error in client.py: {e}")
        sys.exit(1)

# Test 4: Validate server/nle_environment.py
print("\n[4/6] Validating server/nle_environment.py...")
env_path = base_path / "server" / "nle_environment.py"
with open(env_path) as f:
    try:
        tree = ast.parse(f.read())
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        if "NLEEnvironment" in classes:
            print(f"  ✓ Found class: NLEEnvironment")
        else:
            print(f"  ✗ Missing class: NLEEnvironment")
            sys.exit(1)

        # Check for required methods
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "NLEEnvironment":
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                required_methods = ["reset", "step", "state"]
                for method in required_methods:
                    if method in methods:
                        print(f"  ✓ Found method: {method}")
                    else:
                        print(f"  ✗ Missing method: {method}")
                        sys.exit(1)
    except SyntaxError as e:
        print(f"  ✗ Syntax error in server/nle_environment.py: {e}")
        sys.exit(1)

# Test 5: Validate server/app.py
print("\n[5/6] Validating server/app.py...")
app_path = base_path / "server" / "app.py"
with open(app_path) as f:
    content = f.read()
    try:
        tree = ast.parse(content)

        # Check for create_app call
        has_create_app = "create_app" in content
        if has_create_app:
            print(f"  ✓ Found create_app call")
        else:
            print(f"  ✗ Missing create_app call")
            sys.exit(1)

        # Check for app variable
        has_app_var = any(
            isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "app"
                for target in node.targets
            )
            for node in ast.walk(tree)
        )
        if has_app_var:
            print(f"  ✓ Found app variable")
        else:
            print(f"  ✗ Missing app variable")
            sys.exit(1)
    except SyntaxError as e:
        print(f"  ✗ Syntax error in server/app.py: {e}")
        sys.exit(1)

# Test 6: Validate Dockerfile
print("\n[6/6] Validating server/Dockerfile...")
dockerfile_path = base_path / "server" / "Dockerfile"
with open(dockerfile_path) as f:
    content = f.read()

    required_elements = [
        ("FROM", "Base image"),
        ("RUN", "Install commands"),
        ("COPY", "Copy files"),
        ("CMD", "Startup command"),
        ("EXPOSE", "Port exposure"),
    ]

    for element, description in required_elements:
        if element in content:
            print(f"  ✓ Found {description} ({element})")
        else:
            print(f"  ✗ Missing {description} ({element})")
            sys.exit(1)

    # Check for NLE installation
    if "nle" in content.lower():
        print(f"  ✓ Found NLE installation")
    else:
        print(f"  ✗ Missing NLE installation")
        sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("✓ All code structure validations passed!")
print("=" * 70)
print("\nStructure is correct. Integration complete!")
print("\nNext steps:")
print("  1. Build Docker image (from repo root):")
print("     cd /Users/sanyambhutani/GH/OpenEnv")
print("     docker build -f src/envs/nle_env/server/Dockerfile -t nle-env:latest .")
print()
print("  2. Run server:")
print("     docker run -p 8000:8000 nle-env:latest")
print()
print("  3. Test with client (requires OpenEnv dependencies):")
print("     python examples/test_nle.py")
print()
