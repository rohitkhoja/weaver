""" Multi-database manager supporting DuckDB and MySQL """

import duckdb
import pandas as pd
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

from ..config.logging_config import get_logger


logger = get_logger("database.manager")


@dataclass
class QueryResult:
    """Result of a database query."""
    success: bool
    data: Optional[pd.DataFrame] = None
    table_name: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0


class DatabaseManager:
    """
    Multi-database manager supporting DuckDB and MySQL.
    """

    def __init__(self, config_or_connection: Union[str, Any], db_type: str = "duckdb"):
        """
        Initialize database manager.
        
        Args:
            config_or_connection: Database connection string/path or DatabaseConfig object
            db_type: Database type ('duckdb', 'mysql', 'sqlite')
        """
        # Handle both string and config object
        if hasattr(config_or_connection, 'get_connection_string'):
            # It's a DatabaseConfig object
            self.connection_string = config_or_connection.get_connection_string()
            self.db_type = config_or_connection.db_type.lower()
        else:
            # It's a string
            self.connection_string = config_or_connection
            self.db_type = db_type.lower()
            
        self.connection = None
        self.engine = None  # For SQL databases
        self._connect()
        
        logger.info(f"Initialized {self.db_type} database: {self.connection_string}")
    
    def _connect(self) -> None:
        """Establish database connection."""
        try:
            if self.db_type == "duckdb":
                self._connect_duckdb()
            elif self.db_type == "mysql":
                self._connect_mysql()
            elif self.db_type == "sqlite":
                self._connect_sqlite()
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")
                
        except Exception as e:
            logger.error(f"Failed to connect to {self.db_type} database: {e}")
            raise
    
    def _connect_duckdb(self) -> None:
        """Connect to DuckDB."""
        if self.connection_string and self.connection_string != ":memory:":
            # File-based database
            db_path = Path(self.connection_string)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = duckdb.connect(str(db_path))
        else:
            # In-memory database
            self.connection = duckdb.connect()
        
        # Set optimal settings for analytics
        self.connection.execute("SET memory_limit='2GB'")
        self.connection.execute("SET threads=4")
    
    def _connect_mysql(self) -> None:
        """Connect to MySQL."""
        try:
            import sqlalchemy
            from sqlalchemy import create_engine
            
            self.engine = create_engine(
                self.connection_string,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False  # Set to True for SQL debugging
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1"))
                
        except ImportError:
            raise ImportError("SQLAlchemy and PyMySQL are required for MySQL support. "
                            "Install with: pip install sqlalchemy pymysql")
    
    def _connect_sqlite(self) -> None:
        """Connect to SQLite."""
        try:
            import sqlalchemy
            from sqlalchemy import create_engine
            
            self.engine = create_engine(
                self.connection_string,
                pool_pre_ping=True,
                echo=False
            )
            
        except ImportError:
            raise ImportError("SQLAlchemy is required for SQLite support. "
                            "Install with: pip install sqlalchemy")
    
    def table_exists(self, table_name: str) -> bool:
        """Check if table exists."""
        try:
            # Clean table name for consistency with upload_table
            clean_table_name = self._clean_table_name(table_name)
            
            if self.db_type == "duckdb":
                result = self.connection.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_name = ?",
                    [clean_table_name]
                ).fetchone()
                return result is not None
            else:
                # For MySQL/SQLite using SQLAlchemy
                import sqlalchemy
                with self.engine.connect() as conn:
                    if self.db_type == "mysql":
                        result = conn.execute(
                            sqlalchemy.text("SHOW TABLES LIKE :table_name"),
                            {"table_name": clean_table_name}
                        ).fetchone()
                    else:  # sqlite
                        result = conn.execute(
                            sqlalchemy.text("SELECT name FROM sqlite_master WHERE type='table' AND name=:table_name"),
                            {"table_name": clean_table_name}
                        ).fetchone()
                    return result is not None
        except Exception as e:
            logger.error(f"Error checking table existence: {e}")
            return False
    
    def upload_table(self, table_name: str, df: pd.DataFrame, if_exists: str = "replace") -> bool:
        """
        Upload DataFrame to database table.
        
        Args:
            table_name: Name of the table
            df: DataFrame to upload
            if_exists: What to do if table exists ('replace', 'append', 'fail')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            start_time = time.time()
            
            # Clean table name for SQL compatibility
            clean_table_name = self._clean_table_name(table_name)
            
            # Handle existing table
            if self.table_exists(clean_table_name):
                if if_exists == "fail":
                    raise ValueError(f"Table {clean_table_name} already exists")
                elif if_exists == "replace":
                    if self.db_type == "duckdb":
                        self.connection.execute(f"DROP TABLE IF EXISTS {clean_table_name}")
                    else:
                        # Use the same approach as your old database_setup.py
                        import sqlalchemy
                        with self.engine.connect() as conn:
                            conn.execute(sqlalchemy.text(f"DROP TABLE IF EXISTS `{clean_table_name}`"))
                            conn.commit()
                            time.sleep(0.1)  # Small delay like in your old code
            
            if self.db_type == "duckdb":
                # Create table from DataFrame using DuckDB
                self.connection.register(f"temp_{clean_table_name}", df)
                self.connection.execute(f"CREATE TABLE {clean_table_name} AS SELECT * FROM temp_{clean_table_name}")
                self.connection.unregister(f"temp_{clean_table_name}")
            else:
                # Use pandas to_sql for MySQL/SQLite - same as your old code
                df.to_sql(clean_table_name, self.engine, if_exists=if_exists, index=False)
            
            execution_time = time.time() - start_time
            logger.info(f"Uploaded table '{clean_table_name}' with {len(df)} rows, "
                       f"{len(df.columns)} columns in {execution_time:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload table '{table_name}': {e}")
            return False
    
    def execute_query(self, query: str) -> QueryResult:
        """
        Execute SQL query and return results.
        
        Args:
            query: SQL query to execute
            
        Returns:
            QueryResult object with success status and data
        """
        start_time = time.time()
        
        try:
            query_type = query.strip().split()[0].upper()
            
            if self.db_type == "duckdb":
                return self._execute_duckdb_query(query, query_type, start_time)
            else:
                return self._execute_sql_query(query, query_type, start_time)
                
        except Exception as e:
            execution_time = time.time() - start_time
            logger.debug(f"Query execution failed after {execution_time:.2f}s: {e}")
            
            return QueryResult(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    def _execute_duckdb_query(self, query: str, query_type: str, start_time: float) -> QueryResult:
        """Execute query using DuckDB."""
        if query_type == 'SELECT':
            # Execute SELECT query and return DataFrame
            result = self.connection.execute(query).fetchdf()
            execution_time = time.time() - start_time
            
            logger.debug(f"SELECT query executed in {execution_time:.2f}s, "
                       f"returned {len(result)} rows")
            
            return QueryResult(
                success=True,
                data=result,
                execution_time=execution_time
            )
            
        elif query_type == 'CREATE':
            # Execute CREATE TABLE query
            table_name = self._extract_table_name_from_create(query)
            table_name = table_name.lower()
            self.connection.execute(f"DROP TABLE IF EXISTS {table_name}")
            self.connection.execute(query)
            execution_time = time.time() - start_time
            
            logger.info(f"CREATE TABLE executed in {execution_time:.2f}s")
            
            return QueryResult(
                success=True,
                table_name=table_name,
                execution_time=execution_time
            )
            
        else:
            # Execute other queries (INSERT, UPDATE, DELETE, etc.)
            self.connection.execute(query)
            execution_time = time.time() - start_time
            
            logger.info(f"{query_type} query executed in {execution_time:.2f}s")
            
            return QueryResult(
                success=True,
                execution_time=execution_time
            )
    
    def _execute_sql_query(self, query: str, query_type: str, start_time: float) -> QueryResult:
        """Execute query using SQLAlchemy (MySQL/SQLite) - matches your old database_setup.py approach."""
        import sqlalchemy
        
        # Fix MySQL syntax issues
        query = self._fix_mysql_syntax(query)
        
        with self.engine.connect() as conn:
            if query_type == 'SELECT':
                # Execute SELECT query and return DataFrame - same as your old code
                result = conn.execute(sqlalchemy.text(query))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                execution_time = time.time() - start_time
                
                logger.debug(f"SELECT query executed in {execution_time:.2f}s, "
                           f"returned {len(df)} rows")
                
                return QueryResult(
                    success=True,
                    data=df,
                    execution_time=execution_time
                )
                
            elif query_type == 'CREATE':
                # Execute CREATE TABLE query - same approach as your old code
                table_name = self._extract_table_name_from_create(query)
                table_name = table_name.lower()
                
                # Drop existing table if it exists - same as your old code
                conn.execute(sqlalchemy.text(f"DROP TABLE IF EXISTS `{table_name}`"))
                time.sleep(0.1)  # Small delay like in your old code
                conn.execute(sqlalchemy.text(query))
                conn.commit()
                
                execution_time = time.time() - start_time
                logger.info(f"CREATE TABLE executed in {execution_time:.2f}s")
                
                return QueryResult(
                    success=True,
                    table_name=table_name,
                    execution_time=execution_time
                )
                
            else:
                # Execute other queries (INSERT, UPDATE, DELETE, etc.)
                conn.execute(sqlalchemy.text(query))
                conn.commit()
                execution_time = time.time() - start_time
                
                logger.info(f"{query_type} query executed in {execution_time:.2f}s")
                
                return QueryResult(
                    success=True,
                    execution_time=execution_time
                )
    
    def get_table_data(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get all data from a table - matches your old database_setup.py approach.
        
        Args:
            table_name: Name of the table
            limit: Optional limit on number of rows
            
        Returns:
            DataFrame with table data
        """
        try:
            # Clean table name for consistency with upload_table
            clean_table_name = self._clean_table_name(table_name)
            
            if self.db_type == "duckdb":
                query = f"SELECT * FROM {clean_table_name}"
                if limit:
                    query += f" LIMIT {limit}"
                
                result = self.execute_query(query)
                if result.success:
                    return result.data
                else:
                    logger.debug(f"Failed to get table data: {result.error}")
                    return pd.DataFrame()
            else:
                # Use the same approach as your old get_all_rows method
                import sqlalchemy
                with self.engine.connect() as conn:
                    query = f"SELECT * FROM {clean_table_name}"
                    if limit:
                        query += f" LIMIT {limit}"
                    
                    result = conn.execute(sqlalchemy.text(query))
                    conn.commit()
                    
                    df = pd.DataFrame(result.fetchall(), columns=result.keys())
                    return df
                    
        except Exception as e:
            logger.debug(f"Failed to get table data from '{table_name}': {e}")
            return pd.DataFrame()
    
    def list_tables(self) -> List[str]:
        """Get list of all tables in database."""
        try:
            if self.db_type == "duckdb":
                result = self.connection.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
                ).fetchall()
                return [row[0] for row in result]
            else:
                # For MySQL/SQLite
                import sqlalchemy
                with self.engine.connect() as conn:
                    if self.db_type == "mysql":
                        result = conn.execute(sqlalchemy.text("SHOW TABLES")).fetchall()
                    else:  # sqlite
                        result = conn.execute(sqlalchemy.text(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        )).fetchall()
                    return [row[0] for row in result]
        except Exception as e:
            logger.error(f"Failed to list tables: {e}")
            return []
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a table."""
        try:
            # Clean table name for consistency with upload_table
            clean_table_name = self._clean_table_name(table_name)
            
            if self.db_type == "duckdb":
                # Get column information
                columns_info = self.connection.execute(f"DESCRIBE {clean_table_name}").fetchdf()
                
                # Get row count
                row_count = self.connection.execute(f"SELECT COUNT(*) FROM {clean_table_name}").fetchone()[0]
                
                return {
                    "table_name": clean_table_name,
                    "columns": columns_info.to_dict('records'),
                    "row_count": row_count,
                    "column_count": len(columns_info)
                }
            else:
                # For MySQL/SQLite
                import sqlalchemy
                with self.engine.connect() as conn:
                    if self.db_type == "mysql":
                        columns_info = pd.read_sql(f"DESCRIBE {clean_table_name}", conn)
                    else:  # sqlite
                        columns_info = pd.read_sql(f"PRAGMA table_info({clean_table_name})", conn)
                    
                    # Get row count
                    row_count = conn.execute(sqlalchemy.text(f"SELECT COUNT(*) FROM {clean_table_name}")).fetchone()[0]
                    
                    return {
                        "table_name": clean_table_name,
                        "columns": columns_info.to_dict('records'),
                        "row_count": row_count,
                        "column_count": len(columns_info)
                    }
        except Exception as e:
            logger.error(f"Failed to get table info for '{table_name}': {e}")
            return {}
    
    def _clean_table_name(self, table_name: str) -> str:
        """Clean table name for SQL compatibility."""
        # Remove special characters and replace with underscores
        clean_name = re.sub(r'[^\w]', '_', table_name)
        
        # Ensure it doesn't start with a digit
        if clean_name and clean_name[0].isdigit():
            clean_name = f"table_{clean_name}"
        
        # Limit length
        clean_name = clean_name[:64]
        
        # Ensure it's not empty
        if not clean_name:
            clean_name = "table_1"
        
        return clean_name.lower()
    
    def _extract_table_name_from_create(self, query: str) -> Optional[str]:
        """Extract table name from CREATE TABLE query."""
        pattern = r'(?i)\bCREATE\s+TABLE\s+[`"]?([\w]+(?:\.[\w]+)?)[`"]?'
        match = re.search(pattern, query)
        if match:
            return match.group(1).lower()
        return None
    
    def close_connection(self) -> None:
        """Close database connection."""
        if self.db_type == "duckdb" and self.connection:
            self.connection.close()
            self.connection = None
        elif self.engine:
            self.engine.dispose()
            self.engine = None
        logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_connection()
    
    def __del__(self):
        """Destructor to ensure connection is closed."""
        self.close_connection()
    
    def _fix_mysql_syntax(self, query: str) -> str:
        """Fix common MySQL syntax issues compared to SQLite/DuckDB."""
        if self.db_type != "mysql":
            return query
            
        import re
        
        # Fix missing closing parentheses - common issue with generated SQL
        # Count opening and closing parentheses
        open_parens = query.count('(')
        close_parens = query.count(')')
        
        if open_parens > close_parens:
            # Add missing closing parentheses before the semicolon
            missing_parens = open_parens - close_parens
            query = query.rstrip(';').rstrip() + ')' * missing_parens + ';'
        
        # Remove extra parentheses at the end of queries
        query = re.sub(r'\)\s*;\s*$', ';', query.strip())
        
        # Fix concatenation operator: || -> CONCAT()
        # Handle patterns like: 'text' || column || 'text'
        if '||' in query:
            # Find all || concatenations in SELECT clauses
            # This is a simplified fix for the most common cases
            concat_pattern = r"'([^']+)'\s*\|\|\s*(\w+)\s*\|\|\s*'([^']+)'"
            
            def replace_concat(match):
                text1, column, text2 = match.groups()
                return f"CONCAT('{text1}', {column}, '{text2}')"
            
            query = re.sub(concat_pattern, replace_concat, query)
            
            # Handle simpler cases: column || column
            simple_concat_pattern = r"(\w+)\s*\|\|\s*(\w+)"
            query = re.sub(simple_concat_pattern, r"CONCAT(\1, \2)", query)
        
        return query


# Legacy compatibility - maps old DataBase class methods to new DatabaseManager
class DataBase(DatabaseManager):
    """
    Legacy compatibility class that maps old MySQL-based DataBase methods 
    to new DuckDB-based DatabaseManager.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize with DuckDB"""
        super().__init__(db_path)
        logger.warning("DataBase class is deprecated. Use DatabaseManager instead.")
    
    def table_not_exist(self, table_name: str) -> bool:
        """Legacy method - use table_exists instead."""
        return not self.table_exists(table_name)
    
    def execute(self, table_name: str, query: str) -> Tuple[int, Union[pd.DataFrame, str, None]]:
        """
        Legacy method that returns old-style results.
        
        Returns:
            Tuple of (status_code, result) where:
            - status_code: 1 for SELECT, 2 for CREATE, 0 for error
            - result: DataFrame for SELECT, table_name for CREATE, None for error
        """
        result = self.execute_query(query)
        
        if not result.success:
            return 0, None
        
        if result.data is not None:
            return 1, result.data
        elif result.table_name is not None:
            return 2, result.table_name
        else:
            return 1, pd.DataFrame()  # For other successful queries
    
    def get_all_rows(self, table_name: str) -> pd.DataFrame:
        """Legacy method - use get_table_data instead."""
        return self.get_table_data(table_name)
    
    def load_from_sqlite(self, dataset: str) -> Tuple[List[pd.DataFrame], List[str], List[pd.DataFrame]]:
        """
        Legacy method to load data from SQLite files.
        This is replaced by the new DataLoader system, but kept for compatibility.
        """
        # This method is deprecated - the new system uses DataLoader instead
        logger.warning("load_from_sqlite is deprecated. Use DataLoader instead.")
        return [], [], []
