"""
Weaver: Table Question Answering with Embedded Unstructured Columns

A Python package for answering natural language questions on tables with Embedded Unstructured Columns
using LLM-powered planning and execution.
"""

__version__ = "0.1.0"
__author__ = "Rohit Khoja, Devanshu Gupta"
__email__ = "rohit.khoja344@gmail.com"

from .core.weaver import TableQA
from .core.weaver_multi import MultiTableQA
from .config.settings import WeaverConfig

__all__ = [
    "TableQA",
    "MultiTableQA", 
    "WeaverConfig",
]
