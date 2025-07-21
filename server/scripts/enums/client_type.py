"""
Client Type Enumeration
======================

Defines the types of clients supported by the credit scoring system.
"""

from enum import Enum


class ClientType(Enum):
    """Client type enumeration for type-safe operations."""
    NEW = "new"
    RENEWING = "renewing"