"""Simple prompt loading system."""

import os
from pathlib import Path
from typing import Optional
import logging

from .builtin_prompts import get_prompt, DEFAULT_PROMPTS

logger = logging.getLogger(__name__)


class PromptLoader:
    """
    Simple prompt loader:
    1. User external files (if path provided)
    2. Default built-in prompts
    """
    
    def __init__(self, external_prompts_dir: Optional[Path] = None):
        """Initialize prompt loader."""
        self.external_prompts_dir = external_prompts_dir
    
    def load_prompt(self, prompt_type: str, dataset: str = "default") -> str:
        """
        Load a prompt with simple fallback:
        1. User external file (if path provided)
        2. Default built-in prompt
        """
        # 1. Try external file (if directory provided)
        if self.external_prompts_dir:
            external_prompt = self._load_external_prompt(prompt_type, dataset)
            if external_prompt:
                return external_prompt
        
        # 2. Use built-in default prompt
        return get_prompt(prompt_type, dataset)
    
    def _load_external_prompt(self, prompt_type: str, dataset: str) -> Optional[str]:
        """Try to load prompt from external file."""
        if not self.external_prompts_dir or not self.external_prompts_dir.exists():
            return None
        
        # Try dataset-specific file first: prompts_dir/dataset/prompt_type
        dataset_file = self.external_prompts_dir / dataset / prompt_type
        if dataset_file.exists():
            try:
                return dataset_file.read_text(encoding='utf-8').strip()
            except Exception as e:
                logger.warning(f"Failed to load {dataset_file}: {e}")
        
        # Try generic file: prompts_dir/prompt_type
        generic_file = self.external_prompts_dir / prompt_type
        if generic_file.exists():
            try:
                return generic_file.read_text(encoding='utf-8').strip()
            except Exception as e:
                logger.warning(f"Failed to load {generic_file}: {e}")

# Global loader instance
_default_loader: Optional[PromptLoader] = None


def get_default_loader() -> PromptLoader:
    """Get the default prompt loader instance."""
    global _default_loader
    if _default_loader is None:
        _default_loader = PromptLoader()
    return _default_loader


def configure_prompt_loader(external_prompts_dir: Optional[Path] = None):
    """Configure the default prompt loader."""
    global _default_loader
    _default_loader = PromptLoader(external_prompts_dir)


def load_prompt(prompt_type: str, dataset: str = "default") -> str:
    """Load a prompt using the default loader."""
    return get_default_loader().load_prompt(prompt_type, dataset)
