"""Vercel serverless entry point.

Vercel's @vercel/python runtime detects the module-level ASGI `app` and
serves it. All routes (API + static UI) are handled by the single FastAPI
application re-exported here; vercel.json routes every path to this file.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.main import app  # noqa: F401  (Vercel looks for `app`)
