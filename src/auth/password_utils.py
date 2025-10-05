"""
Password hashing and verification utilities
Uses bcrypt for secure password storage
"""

import bcrypt


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt
    BCrypt has a 72-byte limit, passwords are automatically truncated

    Args:
        password: Plain text password

    Returns:
        Hashed password string
    """
    # Encode password and hash with bcrypt
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash

    Args:
        plain_password: Plain text password to verify
        hashed_password: Hashed password to compare against

    Returns:
        True if password matches, False otherwise
    """
    password_bytes = plain_password.encode('utf-8')
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_bytes, hashed_bytes)


def needs_rehash(hashed_password: str) -> bool:
    """
    Check if a password hash needs to be updated
    (e.g., if algorithm or parameters changed)

    Args:
        hashed_password: Existing hashed password

    Returns:
        True if rehash recommended, False otherwise
    """
    return pwd_context.needs_update(hashed_password)
