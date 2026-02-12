#!/usr/bin/env python3
"""
Test deployed CARLA environment (Real Mode).

Verifies:
1. Health endpoint responds
2. Reset works
3. Step with actions works
4. State endpoint works
"""

import requests
import json

BASE_URL = "https://sergiopaniego-carla-env-real.hf.space"

def test_health():
    print("1️⃣ Testing /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"   ✅ Status: {response.status_code}")
        print(f"   Response: {response.json()}")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_reset():
    print("\n2️⃣ Testing /reset endpoint...")
    try:
        response = requests.post(f"{BASE_URL}/reset", timeout=30)
        print(f"   ✅ Status: {response.status_code}")
        data = response.json()
        print(f"   Speed: {data['observation']['speed_kmh']:.1f} km/h")
        print(f"   Lane: {data['observation']['current_lane']}")
        print(f"   Scene: {data['observation']['scene_description'][:80]}...")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_step_observe():
    print("\n3️⃣ Testing /step with observe action...")
    try:
        response = requests.post(
            f"{BASE_URL}/step",
            json={"action": {"action_type": "observe"}},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        print(f"   ✅ Status: {response.status_code}")
        data = response.json()
        print(f"   Speed: {data['observation']['speed_kmh']:.1f} km/h")
        print(f"   Step: {data['observation']['step_number']}")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_step_emergency_stop():
    print("\n4️⃣ Testing /step with emergency_stop action...")
    try:
        response = requests.post(
            f"{BASE_URL}/step",
            json={"action": {"action_type": "emergency_stop"}},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        print(f"   ✅ Status: {response.status_code}")
        data = response.json()
        print(f"   Speed after brake: {data['observation']['speed_kmh']:.1f} km/h")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_state():
    print("\n5️⃣ Testing /state endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/state", timeout=30)
        print(f"   ✅ Status: {response.status_code}")
        data = response.json()
        print(f"   Step count: {data['step_count']}")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    print("=" * 70)
    print("CARLA Environment - Real Mode Deployment Test")
    print("=" * 70)
    print(f"URL: {BASE_URL}\n")

    results = []
    results.append(("Health", test_health()))
    results.append(("Reset", test_reset()))
    results.append(("Step (observe)", test_step_observe()))
    results.append(("Step (emergency_stop)", test_step_emergency_stop()))
    results.append(("State", test_state()))

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")

    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n🎉 All tests passed! Deployment is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the logs above.")

    return all_passed

if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
