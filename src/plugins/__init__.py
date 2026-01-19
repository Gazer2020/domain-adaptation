"""
Plugins package for domain adaptation.

Contains reusable training plugins that can be used across different DA methods.
"""

from plugins.mic import MICPlugin

__all__ = ["MICPlugin"]
