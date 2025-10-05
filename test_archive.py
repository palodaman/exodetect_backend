#!/usr/bin/env python3
"""
Quick test script for archive fetcher
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.archive_fetcher import ArchiveFetcher

def test_koi_fetch():
    """Test fetching KOI data"""
    print("=" * 60)
    print("Testing Archive Fetcher - KOI Query")
    print("=" * 60)

    fetcher = ArchiveFetcher()

    # Test with a known KOI
    test_kois = ["KOI-123.01", "KOI-7.01", "K00070.01"]

    for koi_id in test_kois:
        print(f"\n🔍 Fetching data for {koi_id}...")
        try:
            data = fetcher.fetch_koi_data(koi_id)
            print(f"✅ Success!")
            print(f"   Name: {data['kepoi_name']}")
            print(f"   KepID: {data['kepid']}")
            print(f"   Disposition: {data['koi_disposition']}")
            print(f"   P-Disposition: {data['koi_pdisposition']}")
            print(f"   Score: {data['koi_score']}")
            print(f"   Period: {data['koi_period']} days")
            print(f"   Duration: {data['koi_duration']} hours")
            print(f"   Depth: {data['koi_depth']} ppm")
            print(f"   Vetting Flags: {data['vetting_flags']}")
            return True
        except Exception as e:
            print(f"❌ Failed: {e}")
            continue

    return False

def test_api_endpoint():
    """Test the API endpoint"""
    print("\n" + "=" * 60)
    print("Testing API Endpoint")
    print("=" * 60)

    import requests

    # Test if server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"✅ Server is running")
        print(f"   Response: {response.json()}")

        # Test archive endpoint
        print("\n🔍 Testing archive prediction endpoint...")
        response = requests.post(
            "http://localhost:8000/api/predict/archive",
            json={
                "identifier": "KOI-7.01",
                "mission": "Kepler",
                "include_light_curve": False
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Archive prediction successful!")
            print(f"   Target: {result['target']}")
            print(f"   Probability: {result['model_probability_candidate']:.3f}")
            print(f"   Label: {result['model_label']}")
            print(f"   Confidence: {result['confidence']}")
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            print(f"   Error: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ Server not running. Start with: python src/api/api_server.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("\n🚀 ExoDetect Archive Integration Test\n")

    # Test 1: Archive fetcher
    test1_passed = test_koi_fetch()

    # Test 2: API endpoint (optional if server is running)
    print("\n")
    test2_passed = test_api_endpoint()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Archive Fetcher: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"API Endpoint: {'✅ PASSED' if test2_passed else '⚠️  SKIPPED (server not running)'}")
    print("\n")

    if test1_passed:
        print("🎉 Archive integration is working! You can now:")
        print("   1. Start the backend: python src/api/api_server.py")
        print("   2. Use Explorer Mode in the frontend to search KOIs")
        print("   3. Examples: KOI-7.01, KOI-123.01, KIC-11446443")
    else:
        print("⚠️  Archive fetcher needs attention. Check:")
        print("   1. Internet connection")
        print("   2. NASA Exoplanet Archive availability")
        print("   3. Dependencies: pip install requests pandas numpy")
