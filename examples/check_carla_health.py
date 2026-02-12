#!/usr/bin/env python3
"""
CARLA Environment - Health Check

Quick verification that CARLA environment deployments are running and responding.
Tests health endpoint, web interface, and API docs.

Usage:
    python examples/check_carla_health.py
"""

import requests

def check_health(base_url):
    """Check if CARLA environment is healthy."""

    print(f"🔍 Checking {base_url}...")

    try:
        # Check health endpoint
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"✅ Health endpoint: {response.status_code}")
        print(f"   Response: {response.json()}")

        # Check if it's the web interface
        response = requests.get(f"{base_url}/web", timeout=10)
        print(f"✅ Web interface: {response.status_code}")

        # Check API docs
        response = requests.get(f"{base_url}/docs", timeout=10)
        print(f"✅ API docs: {response.status_code}")

        print(f"\n🎉 Server is UP and responding!")
        print(f"\n📝 Next steps:")
        print(f"   1. Visit web interface: {base_url}/web")
        print(f"   2. Try API docs: {base_url}/docs")
        print(f"   3. Run example: python examples/carla_env_example.py")

        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        print(f"\n⏳ Server might still be starting up...")
        print(f"   Wait a few more seconds and try again")
        return False

if __name__ == "__main__":
    spaces = {
        "Mock Mode": "https://sergiopaniego-carla-env.hf.space",
        "Real Mode": "https://sergiopaniego-carla-env-real.hf.space",
    }

    print("=" * 60)
    print("CARLA Environment Health Check")
    print("=" * 60)

    for name, url in spaces.items():
        print(f"\n{name}:")
        print("-" * 60)
        check_health(url)
        print()
