"""Configuration settings for Weaver."""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union
from pathlib import Path


@dataclass
class LLMConfig:
    """LLM configuration for LiteLLM.
    
    Set your model using LiteLLM format with provider/model:
    
    Examples:
        export LLM_MODEL="openai/gpt-4o-mini"
        export LLM_MODEL="anthropic/claude-3-haiku"  
        export LLM_MODEL="gemini/gemini-2.0-flash-exp"
        export LLM_MODEL="azure/your-deployment-name"
        
    LiteLLM automatically handles API keys from environment variables.
    See LiteLLM documentation for provider-specific setup.
    """
    
    model: str = "openai/gpt-4o-mini"
    temperature: float = 0.01
    max_tokens: int = 2048


@dataclass 
class DatabaseConfig:
    """Database configuration."""
    
    db_type: str = "duckdb"  # duckdb, mysql, postgres, sqlite
    db_path: str = "weaver_data.db"  # for duckdb/sqlite
    db_url: Optional[str] = None  # for postgres: postgresql://user:pass@host:port/db
    
    # MySQL-specific settings
    db_host: str = "localhost"
    db_port: int = 3306
    db_name: str = "weaver"
    db_user: str = "root"
    db_password: str = ""
    
    def get_connection_string(self) -> str:
        """Get database connection string."""
        if self.db_url:
            return self.db_url
        elif self.db_type == "duckdb":
            return self.db_path
        elif self.db_type == "sqlite":
            return f"sqlite:///{self.db_path}"
        elif self.db_type == "mysql":
            return f"mysql+pymysql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")


@dataclass
class WeaverConfig:
    """Main configuration for Weaver QA system."""
    
    # LLM Configuration
    llm: LLMConfig = field(default_factory=LLMConfig)
    
    # Database Configuration  
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # File paths
    results_dir: Path = field(default_factory=lambda: Path("results"))
    datasets_dir: Path = field(default_factory=lambda: Path("datasets"))
    prompts_dir: Path = field(default_factory=lambda: Path("prompts"))
    
    # Processing settings
    parallel_processes: int = 1
    default_dataset: str = "wikitq"
    
    # Table processing
    filter_relevant_columns: bool = True
    max_table_size: int = 10000  # max rows to process
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Ensure directories exist
        for dir_path in [self.results_dir, self.datasets_dir, self.prompts_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls, **kwargs) -> "WeaverConfig":
        """Create config from environment variables and .env file."""
        # Load .env file first
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        
        config = cls(**kwargs)
        
        # Override LLM settings from environment
        if os.getenv("LLM_MODEL"):
            config.llm.model = os.getenv("LLM_MODEL")
        if os.getenv("LLM_TEMPERATURE"):
            config.llm.temperature = float(os.getenv("LLM_TEMPERATURE"))
        if os.getenv("LLM_MAX_TOKENS"):
            config.llm.max_tokens = int(os.getenv("LLM_MAX_TOKENS"))
            
        # Override database settings
        if os.getenv("WEAVER_DB_TYPE"):
            config.database.db_type = os.getenv("WEAVER_DB_TYPE")
        if os.getenv("WEAVER_DB_PATH"):
            config.database.db_path = os.getenv("WEAVER_DB_PATH")
        if os.getenv("DATABASE_URL"):
            config.database.db_url = os.getenv("DATABASE_URL")
        
        # MySQL-specific settings
        if os.getenv("WEAVER_DB_HOST"):
            config.database.db_host = os.getenv("WEAVER_DB_HOST")
        if os.getenv("WEAVER_DB_PORT"):
            config.database.db_port = int(os.getenv("WEAVER_DB_PORT"))
        if os.getenv("WEAVER_DB_NAME"):
            config.database.db_name = os.getenv("WEAVER_DB_NAME")
        if os.getenv("WEAVER_DB_USER"):
            config.database.db_user = os.getenv("WEAVER_DB_USER")
        if os.getenv("WEAVER_DB_PASSWORD"):
            config.database.db_password = os.getenv("WEAVER_DB_PASSWORD")
            
        # Override directory paths
        if os.getenv("WEAVER_RESULTS_DIR"):
            config.results_dir = Path(os.getenv("WEAVER_RESULTS_DIR"))
        if os.getenv("WEAVER_DATASETS_DIR"):
            config.datasets_dir = Path(os.getenv("WEAVER_DATASETS_DIR"))
        if os.getenv("WEAVER_PROMPTS_DIR"):
            config.prompts_dir = Path(os.getenv("WEAVER_PROMPTS_DIR"))
            
        # Override other settings
        if os.getenv("WEAVER_LOG_LEVEL"):
            config.log_level = os.getenv("WEAVER_LOG_LEVEL")
        if os.getenv("WEAVER_PARALLEL_PROCESSES"):
            config.parallel_processes = int(os.getenv("WEAVER_PARALLEL_PROCESSES"))
            
        return config
    
    def validate(self):
        """Validate configuration settings."""
        if self.parallel_processes < 1:
            raise ValueError("parallel_processes must be at least 1")
        
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise ValueError(f"Invalid log_level: {self.log_level}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "llm": {
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens
            },
            "database": {
                "db_type": self.database.db_type,
                "db_path": self.database.db_path
            },
            "results_dir": str(self.results_dir),
            "datasets_dir": str(self.datasets_dir),
            "prompts_dir": str(self.prompts_dir),
            "parallel_processes": self.parallel_processes,
            "log_level": self.log_level
        }
