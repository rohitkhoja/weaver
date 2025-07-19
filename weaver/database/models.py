"""Database models and metadata classes."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import pandas as pd


@dataclass
class ColumnMetadata:
    """Metadata for a database column."""
    name: str
    data_type: str
    is_nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    description: Optional[str] = None
    

@dataclass 
class TableMetadata:
    """Metadata for a database table."""
    name: str
    columns: List[ColumnMetadata]
    row_count: int = 0
    description: Optional[str] = None
    
    @classmethod
    def from_dataframe(cls, name: str, df: pd.DataFrame, description: Optional[str] = None) -> "TableMetadata":
        """Create table metadata from a DataFrame."""
        columns = []
        
        for col_name in df.columns:
            # Infer data type
            dtype = str(df[col_name].dtype)
            if dtype.startswith('int'):
                data_type = 'INTEGER'
            elif dtype.startswith('float'):
                data_type = 'DOUBLE'
            elif dtype == 'bool':
                data_type = 'BOOLEAN'
            else:
                data_type = 'VARCHAR'
            
            # Check for null values
            is_nullable = df[col_name].isnull().any()
            
            columns.append(ColumnMetadata(
                name=col_name,
                data_type=data_type,
                is_nullable=is_nullable
            ))
        
        return cls(
            name=name,
            columns=columns,
            row_count=len(df),
            description=description
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'columns': [
                {
                    'name': col.name,
                    'data_type': col.data_type,
                    'is_nullable': col.is_nullable,
                    'is_primary_key': col.is_primary_key,
                    'is_foreign_key': col.is_foreign_key,
                    'description': col.description
                }
                for col in self.columns
            ],
            'row_count': self.row_count,
            'description': self.description
        }
