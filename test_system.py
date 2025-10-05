#!/usr/bin/env python3
"""
Comprehensive system test for ExoDetect backend
Tests database, authentication, and API endpoints
"""

import asyncio
import sys
import os
import requests
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test results tracker
test_results = {
    "passed": [],
    "failed": [],
    "warnings": []
}


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_test(name, status, message=""):
    """Print test result"""
    icon = "✅" if status == "pass" else "❌" if status == "fail" else "⚠️"
    print(f"{icon} {name}")
    if message:
        print(f"   → {message}")

    if status == "pass":
        test_results["passed"].append(name)
    elif status == "fail":
        test_results["failed"].append(name)
    else:
        test_results["warnings"].append(name)


async def test_database_connection():
    """Test MongoDB connection"""
    print_section("Testing Database Connection")

    try:
        from database import init_db, close_db, User, Organization

        # Initialize database
        await init_db()
        print_test("Database initialization", "pass", "Connected to MongoDB")

        # Test document creation
        try:
            # Create test organization
            org = Organization(name=f"Test Org {datetime.now().timestamp()}")
            await org.insert()
            print_test("Create organization", "pass", f"Org ID: {org.id}")

            # Create test user
            from auth import hash_password
            user = User(
                email=f"test{datetime.now().timestamp()}@example.com",
                username=f"testuser{int(datetime.now().timestamp())}",
                password_hash=hash_password("testpassword"),
                organization_id=str(org.id)
            )
            await user.insert()
            print_test("Create user", "pass", f"User ID: {user.id}")

            # Clean up
            await user.delete()
            await org.delete()
            print_test("Database cleanup", "pass", "Test data removed")

        except Exception as e:
            print_test("Database operations", "fail", str(e))

        await close_db()

    except Exception as e:
        print_test("Database connection", "fail", str(e))
        print("⚠️  Make sure MongoDB is running:")
        print("   docker run -d -p 27017:27017 --name mongodb mongo:latest")


async def test_authentication():
    """Test authentication system"""
    print_section("Testing Authentication System")

    try:
        from auth import (
            hash_password,
            verify_password,
            create_access_token,
            decode_token,
            create_oauth_compatible_token
        )

        # Test password hashing
        password = "testpassword123"
        hashed = hash_password(password)
        if verify_password(password, hashed):
            print_test("Password hashing", "pass", "BCrypt working correctly")
        else:
            print_test("Password hashing", "fail", "Password verification failed")

        # Test JWT tokens
        token_data = {"sub": "user123", "email": "test@example.com"}
        token = create_access_token(token_data)
        decoded = decode_token(token)

        if decoded and decoded.get("sub") == "user123":
            print_test("JWT token creation", "pass", "Tokens encode/decode correctly")
        else:
            print_test("JWT token creation", "fail", "Token decoding failed")

        # Test OAuth-compatible tokens
        oauth_tokens = create_oauth_compatible_token(
            user_id="user123",
            email="test@example.com",
            provider="internal"
        )

        if all(k in oauth_tokens for k in ["access_token", "refresh_token", "token_type"]):
            print_test("OAuth-compatible tokens", "pass", "All token types present")
        else:
            print_test("OAuth-compatible tokens", "fail", "Missing token fields")

    except Exception as e:
        print_test("Authentication system", "fail", str(e))


def test_api_server():
    """Test API server startup and endpoints"""
    print_section("Testing API Server")

    base_url = "http://localhost:8000"

    # Test if server is running
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print_test("API server running", "pass", f"Status: {response.status_code}")
        else:
            print_test("API server running", "warning", "Server responded but not 200")

    except requests.ConnectionError:
        print_test("API server running", "fail", "Server not accessible")
        print("⚠️  Start the server with: python src/api/api_server.py")
        return

    except Exception as e:
        print_test("API server running", "fail", str(e))
        return

    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print_test("Health endpoint", "pass", response.json().get("status", ""))
        else:
            print_test("Health endpoint", "fail", f"Status: {response.status_code}")
    except Exception as e:
        print_test("Health endpoint", "fail", str(e))

    # Test OAuth providers endpoint
    try:
        response = requests.get(f"{base_url}/api/auth/oauth/providers", timeout=5)
        if response.status_code == 200:
            data = response.json()
            providers = len(data.get("available_providers", []))
            print_test("OAuth providers endpoint", "pass", f"{providers} providers listed")
        else:
            print_test("OAuth providers endpoint", "fail", f"Status: {response.status_code}")
    except Exception as e:
        print_test("OAuth providers endpoint", "fail", str(e))


def test_model_loading():
    """Test model loading"""
    print_section("Testing Model Loading")

    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        from ml.exoplanet_predictor import ExoplanetPredictor

        # Try to load default model
        predictor = ExoplanetPredictor(model_name="xgboost_enhanced")
        if predictor.model is not None:
            print_test("Model loading", "pass", f"Loaded: {predictor.model_name}")
            print_test("Model path", "pass", f"Dir: {predictor.model_dir}")
        else:
            print_test("Model loading", "fail", "Model is None")

    except Exception as e:
        print_test("Model loading", "fail", str(e))


def test_dependencies():
    """Test all required dependencies"""
    print_section("Testing Dependencies")

    dependencies = [
        ("fastapi", "FastAPI"),
        ("pydantic", "Pydantic"),
        ("motor", "Motor (MongoDB)"),
        ("beanie", "Beanie ODM"),
        ("jose", "Python-JOSE"),
        ("passlib", "Passlib"),
        ("email_validator", "Email Validator"),
        ("dotenv", "Python-dotenv"),
        ("apscheduler", "APScheduler"),
    ]

    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print_test(display_name, "pass", f"Module: {module_name}")
        except ImportError:
            print_test(display_name, "fail", f"Missing: {module_name}")


def print_summary():
    """Print test summary"""
    print_section("Test Summary")

    total = len(test_results["passed"]) + len(test_results["failed"]) + len(test_results["warnings"])
    passed = len(test_results["passed"])
    failed = len(test_results["failed"])
    warnings = len(test_results["warnings"])

    print(f"\nTotal Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Warnings: {warnings}")

    if failed > 0:
        print("\n❌ Failed Tests:")
        for test in test_results["failed"]:
            print(f"   - {test}")

    if warnings > 0:
        print("\n⚠️  Warnings:")
        for test in test_results["warnings"]:
            print(f"   - {test}")

    print("\n" + "="*70)

    if failed == 0:
        print("🎉 All critical tests passed! System is ready.")
    else:
        print("⚠️  Some tests failed. Please fix issues before deploying.")

    print("="*70 + "\n")


async def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*20 + "ExoDetect System Tests" + " "*25 + "║")
    print("╚" + "="*68 + "╝")

    # Run async tests
    await test_database_connection()
    await test_authentication()

    # Run sync tests
    test_dependencies()
    test_model_loading()
    test_api_server()

    # Print summary
    print_summary()


if __name__ == "__main__":
    asyncio.run(main())
