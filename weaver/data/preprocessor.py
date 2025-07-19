"""Table preprocessing utilities"""

import pandas as pd
import re
from typing import List, Optional, Tuple, Set
from sqlalchemy.dialects.mysql import dialect

from ..config.logging_config import get_logger


logger = get_logger("data.preprocessor")


class TablePreprocessor:
    """Preprocessor for cleaning and preparing tables for QA."""
    
    def __init__(self, max_column_width: int = 100, max_rows: int = 1000):
        """
        Initialize preprocessor.
        
        Args:
            max_column_width: Maximum width for column values
            max_rows: Maximum number of rows to process
        """
        self.max_column_width = max_column_width
        self.max_rows = max_rows
        # SQL keywords from SQLAlchemy MySQL dialect - comprehensive list
        self.sql_keywords = set(dialect().preparer.reserved_words)
    
    def clean_table(self, table: pd.DataFrame, table_name: str) -> Tuple[str, pd.DataFrame]:
        """
        Clean table for SQL compatibility.
        
        Args:
            table: Input DataFrame
            table_name: Name of the table
            
        Returns:
            Tuple of (cleaned_table_name, cleaned_dataframe)
        """
        logger.debug(f"Cleaning table: {table_name}")
        
        # Clean table name
        clean_table_name = self._clean_table_name(table_name)
        
        # Clean column names
        cleaned_table = table.copy()
        cleaned_table.columns = self._clean_column_names(cleaned_table.columns)
        
        # Handle null values
        cleaned_table = cleaned_table.where(pd.notnull(cleaned_table), None)
        
        # Limit rows if necessary
        if len(cleaned_table) > self.max_rows:
            logger.warning(f"Table {clean_table_name} has {len(cleaned_table)} rows, limiting to {self.max_rows}")
            cleaned_table = cleaned_table.head(self.max_rows)
        
        # Clean cell values
        cleaned_table = self._clean_cell_values(cleaned_table)
        
        logger.debug(f"Cleaned table {clean_table_name}: {cleaned_table.shape}")
        return clean_table_name, cleaned_table
    
    def _clean_table_name(self, table_name: str) -> str:
        """Clean table name for SQL compatibility."""
        # Remove special characters
        clean_name = re.sub(r'[^\w\s]', '', table_name)
        
        # Replace spaces with underscores
        clean_name = re.sub(r'\s+', '_', clean_name)
        
        # Ensure it doesn't start with a digit
        if clean_name and clean_name[0].isdigit():
            clean_name = f"table_{clean_name}"
        
        # Limit length
        clean_name = clean_name[:64]
        
        # Ensure it's not empty
        if not clean_name:
            clean_name = "table_1"
        
        return clean_name
    
    def _clean_column_names(self, columns: pd.Index) -> List[str]:
        """Clean column names for SQL compatibility."""
        clean_columns = []
        
        for col in columns:
            # Convert to string and clean
            clean_col = str(col) if col is not None else "column"
            clean_col = re.sub(r'[^\w\s]', '', clean_col)
            clean_col = re.sub(r'\s+', '_', clean_col)
            
            # Ensure it's not empty
            if not clean_col:
                clean_col = f"column_{len(clean_columns)}"
            
            clean_columns.append(clean_col)
        
        # Handle SQL keywords and duplicates
        clean_columns = self._rename_sql_keywords(clean_columns)
        clean_columns = self._handle_duplicate_columns(clean_columns)
        
        return clean_columns
    
    def _rename_sql_keywords(self, columns: List[str]) -> List[str]:
        """Rename columns that are SQL keywords."""
        new_columns = []
        
        for col in columns:
            if col.lower() in self.sql_keywords:
                new_col = f"{col}_1"
                counter = 1
                while new_col.upper() in self.sql_keywords or new_col in new_columns:
                    counter += 1
                    new_col = f"{col}_{counter}"
                new_columns.append(new_col)
            else:
                new_columns.append(col)
        
        return new_columns
    
    def _handle_duplicate_columns(self, columns: List[str]) -> List[str]:
        """Handle duplicate column names."""
        seen = set()
        new_columns = []
        
        for col in columns:
            if col in seen:
                counter = 1
                new_col = f"{col}_{counter}"
                while new_col in seen:
                    counter += 1
                    new_col = f"{col}_{counter}"
                new_columns.append(new_col)
                seen.add(new_col)
            else:
                new_columns.append(col)
                seen.add(col)
        
        return new_columns
    
    def _clean_cell_values(self, table: pd.DataFrame) -> pd.DataFrame:
        """Clean cell values in the table."""
        cleaned_table = table.copy()
        
        for col in cleaned_table.columns:
            if cleaned_table[col].dtype == 'object':
                # Clean string values
                cleaned_table[col] = cleaned_table[col].apply(self._clean_string_value)
        
        return cleaned_table
    
    def _clean_string_value(self, value) -> Optional[str]:
        """Clean individual string values."""
        if pd.isna(value) or value is None:
            return None
        
        # Convert to string
        str_value = str(value).strip()
        
        # Handle empty strings
        if not str_value:
            return None
        
        # Limit length
        if len(str_value) > self.max_column_width:
            str_value = str_value[:self.max_column_width-3] + "..."
        
        return str_value
    
    def filter_relevant_columns(
        self, 
        table: pd.DataFrame, 
        question: str, 
        column_descriptions: Optional[str] = None,
        llm_client = None,
        paragraphs: Optional[str] = None,
        table_name: str = "table"
    ) -> pd.DataFrame:
        """
        Filter table to only include columns relevant to the question using LLM.
        
        Args:
            table: Input DataFrame
            question: Question being asked
            column_descriptions: Optional column descriptions  
            llm_client: LLM client for column relevance detection
            paragraphs: Additional context paragraphs
            table_name: Name of the table for context
            
        Returns:
            Filtered DataFrame with only relevant columns
        """
        if llm_client is None:
            # Fallback to simple keyword matching
            logger.debug("No LLM client provided, unable to filter columns")
            logger.debug("Returning original table without column filtering")
            return table
        
        try:
            logger.info(f"Filtering relevant columns for question: {question[:100]}...")
            logger.debug(f"Table shape: {table.shape}")
            logger.debug(f"Original columns: {list(table.columns)}")
            
            # Create prompt to get relevant columns using LLM
            prompt = f'''
            Given column descriptions, Table and Question return a list of columns that can be relevant to solving the question (even if slightly relevant):
            
            Table name: {table_name}
            Table columns: {list(table.columns)}
            Table preview:
            {table.head().to_html()}
            Question: {question}
            Additional information: {paragraphs or "No additional information"}
            Column descriptions: {column_descriptions or "No descriptions provided"}

            Example output: ['Column_1', 'Column_2']
            
            Instructions:
            1. Do not provide any explanations, just give the columns as a Python list
            2. The list will be used to filter the table dataframe directly so ensure column names match exactly
         

            Output:
            '''
            
            logger.debug("Calling LLM for column filtering...")
            # Get LLM response  
            llm_response = llm_client.call(prompt)
            logger.debug(f"LLM column filter response: '{llm_response}'")
            logger.debug(f"LLM response length: {len(llm_response)}")
            
            # Parse the response to get column list
            try:
                # Clean the response first
                clean_response = llm_response.strip()
                
                # Try to find a list pattern in the response
                import re
                list_pattern = r'\[([^\]]+)\]'
                list_match = re.search(list_pattern, clean_response)
                
                if list_match:
                    # Extract the content inside brackets
                    list_content = list_match.group(1)
                    # Split by comma and clean each item
                    relevant_cols = []
                    for item in list_content.split(','):
                        item = item.strip().strip("'\"")
                        if item:
                            relevant_cols.append(item)
                    logger.debug(f"Parsed columns from list pattern: {relevant_cols}")
                else:
                    # Try to evaluate as Python list
                    relevant_cols = eval(clean_response)
                    if not isinstance(relevant_cols, list):
                        raise ValueError("Response is not a list")
                    logger.debug(f"Parsed LLM response as list: {relevant_cols}")
                
            except Exception as e:
                logger.warning(f"Failed to parse LLM response as list: {e}")
                logger.debug(f"Raw response: '{llm_response}'")
                logger.debug("Attempting fallback parsing...")
                # Fallback parsing - extract column names from response
                relevant_cols = []
                for col in table.columns:
                    if col in llm_response:
                        relevant_cols.append(col)
                logger.debug(f"Fallback parsed columns: {relevant_cols}")
            
            # Validate columns exist in table
            final_columns = []
            for col in relevant_cols:
                if col in table.columns:
                    final_columns.append(col)
                    logger.debug(f"✓ Column '{col}' found in table")
                else:
                    logger.warning(f"✗ Column '{col}' not found in table")
            
            # If no relevant columns found, keep all columns
            if not final_columns:
                logger.warning("No relevant columns identified, keeping all columns")
                logger.debug("Using all columns as fallback")
                final_columns = list(table.columns)
            
            logger.info(f"Filtered from {len(table.columns)} to {len(final_columns)} columns: {final_columns}")
            logger.debug(f"Column filtering complete. Final table shape: {table[final_columns].shape}")
            return table[final_columns]
            
        except Exception as e:
            logger.error(f"Error in LLM-based column filtering: {e}")
            logger.debug(f"Exception type: {type(e)}")
            logger.debug(f"Exception details: {str(e)}")
            logger.debug("Falling back to original table without filtering")
            return table

    
    def normalize_data_types(self, table: pd.DataFrame) -> pd.DataFrame:
        """Normalize data types in the table."""
        normalized_table = table.copy()
        
        for col in normalized_table.columns:
            # Try to convert to numeric if possible
            try:
                # Check if the column contains numeric values
                numeric_series = pd.to_numeric(normalized_table[col], errors='coerce')
                if not numeric_series.isna().all():
                    # If more than 50% of values are numeric, convert the column
                    numeric_ratio = numeric_series.notna().sum() / len(normalized_table)
                    if numeric_ratio > 0.5:
                        normalized_table[col] = numeric_series
            except:
                # Keep as object type
                pass
        
        return normalized_table
