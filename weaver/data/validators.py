"""Data validation utilities."""

from typing import Any, Dict, List, Optional
import pandas as pd

from ..config.logging_config import get_logger


logger = get_logger("data.validators")


class DataValidator:
    """Validator for dataset and table data."""
    
    def validate_dataset(self, data: List[Dict[str, Any]], dataset_name: str) -> bool:
        """
        Validate dataset structure.
        
        Args:
            data: List of dataset items
            dataset_name: Name of the dataset
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        if not data:
            raise ValueError("Dataset is empty")
        
        required_fields = self._get_required_fields(dataset_name)
        
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Item {i} is not a dictionary")
            
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"Item {i} missing required field: {field}")
                
                if not item[field]:  # Check for empty values
                    logger.warning(f"Item {i} has empty value for field: {field}")
        
        logger.info(f"Validated {len(data)} items for dataset: {dataset_name}")
        return True
    
    def validate_table(self, table: pd.DataFrame, table_name: str) -> bool:
        """
        Validate table structure.
        
        Args:
            table: DataFrame to validate
            table_name: Name of the table
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        if table.empty:
            raise ValueError(f"Table {table_name} is empty")
        
        if table.shape[0] == 0:
            raise ValueError(f"Table {table_name} has no rows")
        
        if table.shape[1] == 0:
            raise ValueError(f"Table {table_name} has no columns")
        
        # Check for valid column names
        invalid_columns = []
        for col in table.columns:
            if not isinstance(col, str) or not col.strip():
                invalid_columns.append(col)
        
        if invalid_columns:
            raise ValueError(f"Table {table_name} has invalid column names: {invalid_columns}")
        
        # Check for duplicate column names
        if len(table.columns) != len(set(table.columns)):
            duplicates = table.columns[table.columns.duplicated()].tolist()
            raise ValueError(f"Table {table_name} has duplicate column names: {duplicates}")
        
        logger.debug(f"Validated table {table_name}: {table.shape[0]} rows, {table.shape[1]} columns")
        return True
    
    def validate_question(self, question: str) -> bool:
        """
        Validate question format.
        
        Args:
            question: Question string
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        if not question or not question.strip():
            raise ValueError("Question is empty")
        
        if len(question) > 1000:
            raise ValueError("Question is too long (max 1000 characters)")
        
        return True
    
    def _get_required_fields(self, dataset_name: str) -> List[str]:
        """Get required fields for a dataset."""
        base_fields = ['question', 'table_file_name']
        
        dataset_specific = {
            'wikitq': ['table_id', 'target_value'],
            'tabfact': ['table_id', 'label'],
            'finqa': ['table_name', 'target_value'],
            'ott-qa': ['table_name', 'target_value']
        }
        
        return base_fields + dataset_specific.get(dataset_name, [])
    
    def check_data_quality(self, table: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data quality metrics for a table.
        
        Args:
            table: DataFrame to analyze
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {
            'total_rows': len(table),
            'total_columns': len(table.columns),
            'missing_values': table.isnull().sum().sum(),
            'duplicate_rows': table.duplicated().sum(),
            'empty_strings': 0,
            'column_types': table.dtypes.to_dict()
        }
        
        # Count empty strings
        for col in table.select_dtypes(include=['object']):
            metrics['empty_strings'] += (table[col] == '').sum()
        
        # Calculate missing percentage
        total_cells = metrics['total_rows'] * metrics['total_columns']
        metrics['missing_percentage'] = (metrics['missing_values'] / total_cells * 100) if total_cells > 0 else 0
        
        return metrics
