"""Base classes and interfaces for Weaver QA systems."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from ..config.settings import WeaverConfig


@dataclass
class QAResult:
    """Result of a question answering operation."""
    
    question: str
    answer: str
    plan: Optional[str] = None
    sql_code: Optional[str] = None
    execution_log: Optional[List[str]] = None
    is_correct: Optional[bool] = None
    gold_answer: Optional[str] = None
    table_id: Optional[str] = None
    token_stats: Optional[Dict[str, Any]] = None  # Token usage statistics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "plan": self.plan,
            "sql_code": self.sql_code,
            "execution_log": self.execution_log,
            "is_correct": self.is_correct,
            "gold_answer": self.gold_answer,
            "table_id": self.table_id,
            "token_stats": self.token_stats
        }


class BaseQA(ABC):
    """Abstract base class for QA systems."""
    
    def __init__(self, config: WeaverConfig):
        """Initialize the QA system with configuration."""
        self.config = config
        self.config.validate()
        
        # Initialize components
        self._setup_logging()
        self._setup_database()
        self._setup_llm()
    
    def _setup_logging(self) -> None:
        """Setup logging for the QA system using config values."""
        from ..config.logging_config import setup_logging
        setup_logging(
            log_level=self.config.log_level,
            log_file=self.config.log_file,
            enable_console=True
        )
    
    @abstractmethod
    def _setup_database(self) -> None:
        """Setup database connection."""
        pass
    
    @abstractmethod
    def _setup_llm(self) -> None:
        """Setup LLM client."""
        pass
    
    
    def validate_input(self, question: str) -> bool:
        """Validate input question."""
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        if len(question) > 1000:  # Reasonable limit
            raise ValueError("Question too long (max 1000 characters)")
        
        return True
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if hasattr(self, 'database') and self.database:
            self.database.close_connection()


class TableProcessor(ABC):
    """Abstract base class for table processing."""
    
    @abstractmethod
    def preprocess_table(self, table: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Preprocess a table for QA."""
        pass
    
    @abstractmethod
    def filter_relevant_columns(self, table: pd.DataFrame, question: str) -> pd.DataFrame:
        """Filter table to only relevant columns."""
        pass
