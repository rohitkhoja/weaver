"""Data loader"""

import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from abc import ABC, abstractmethod

from ..config.logging_config import get_logger
from .validators import DataValidator


logger = get_logger("data.loader")


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    @abstractmethod
    def load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load dataset from file."""
        pass
    
    @abstractmethod
    def get_table(self, item: Dict[str, Any]) -> pd.DataFrame:
        """Get table from dataset item."""
        pass


class WikiTQLoader(BaseDatasetLoader):
    """Loader for WikiTableQuestions dataset."""
    
    def load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load WikiTQ data from JSON file."""
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_table(self, item: Dict[str, Any]) -> pd.DataFrame:
        """Load table for WikiTQ item."""
        table_path = item['table_file_name']
        
        # Parse context to get actual file path
        context_parts = self._parse_context(table_path)
        csv_path = f'./WikiTableQuestions/csv/{context_parts[0]}-csv/{context_parts[1]}.csv'
        
        try:
            return pd.read_csv(csv_path, header=0)
        except FileNotFoundError:
            # Try .table format
            table_path = f'./WikiTableQuestions/csv/{context_parts[0]}-csv/{context_parts[1]}.table'
            table = pd.read_table(table_path, sep='|')
            table.columns = table.columns.str.strip()
            return table.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    def _parse_context(self, context: str) -> Tuple[str, str]:
        """Parse WikiTQ context string."""
        import re
        match = re.match(r'csv/(\d+)-csv/(\d+).csv$', context)
        if not match:
            raise ValueError(f"Invalid context format: {context}")
        return match.groups()


class TabFactLoader(BaseDatasetLoader):
    """Loader for TabFact dataset."""
    
    def load_data(self, data_path: str) -> List[Dict[str, Any]]:
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_table(self, item: Dict[str, Any]) -> pd.DataFrame:
        table_path = f'TabFact/{item["table_file_name"]}'
        return pd.read_csv(table_path, header=0)


class FinQALoader(BaseDatasetLoader):
    """Loader for FinQA dataset."""
    
    def load_data(self, data_path: str) -> List[Dict[str, Any]]:
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_table(self, item: Dict[str, Any]) -> pd.DataFrame:
        table_path = f'FinQA/{item["table_file_name"]}'
        return pd.read_csv(table_path, header=0)


class OTTQALoader(BaseDatasetLoader):
    """Loader for OTT-QA dataset."""
    
    def load_data(self, data_path: str) -> List[Dict[str, Any]]:
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_table(self, item: Dict[str, Any]) -> pd.DataFrame:
        """Load and process OTT-QA table with request data."""
        table_file = item["table_file_name"]
        table_path = f'OTT-QA/data/traindev_tables_tok/{table_file}.json'
        
        # Load table JSON
        with open(table_path, 'r', encoding='utf-8') as f:
            table_json = json.load(f)
        
        # Extract headers and data
        headers = [header[0] for header in table_json['header']]
        df_data = {header: [] for header in headers}
        
        # Process each row
        for row in table_json['data']:
            for col_idx, cell in enumerate(row):
                col_name = headers[col_idx]
                cell_text = cell[0]
                
                # Process links if they exist
                if len(cell) > 1 and cell[1]:
                    try:
                        request_path = f'OTT-QA/data/traindev_request_tok/{table_file}.json'
                        with open(request_path, 'r', encoding='utf-8') as f:
                            request_json = json.load(f)
                        
                        expanded_texts = []
                        for link in cell[1]:
                            if link in request_json:
                                expanded_texts.append(request_json[link])
                        
                        if expanded_texts:
                            cell_text += " | " + " | ".join(expanded_texts)
                    except Exception as e:
                        logger.warning(f"Error loading request data for {table_file}: {e}")
                
                df_data[col_name].append(cell_text)
        
        # Ensure all columns have same length
        max_length = max(len(values) for values in df_data.values())
        for col in df_data:
            current_length = len(df_data[col])
            if current_length < max_length:
                df_data[col].extend([None] * (max_length - current_length))
        
        return pd.DataFrame(df_data)


class DataLoader:
    """Main data loader class that handles multiple dataset formats."""
    
    LOADERS = {
        'wikitq': WikiTQLoader,
        'tabfact': TabFactLoader,
        'finqa': FinQALoader,
        'ott-qa': OTTQALoader
    }
    
    def __init__(self, dataset_name: str, data_dir: Optional[Path] = None):
        """
        Initialize data loader.
        
        Args:
            dataset_name: Name of the dataset
            data_dir: Directory containing dataset files
        """
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir or Path('./datasets')
        self.validator = DataValidator()
        
        if self.dataset_name not in self.LOADERS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        self.loader = self.LOADERS[self.dataset_name]()
        self.data: List[Dict[str, Any]] = []
        self.current_index = 0
        
        logger.info(f"Initialized data loader for dataset: {dataset_name}")
    
    def load_dataset(self, data_path: Optional[str] = None) -> None:
        """Load dataset from file."""
        if data_path is None:
            data_path = self.data_dir / f"{self.dataset_name}.json"
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        
        logger.info(f"Loading dataset from: {data_path}")
        self.data = self.loader.load_data(data_path)
        
        # Validate loaded data
        self.validator.validate_dataset(self.data, self.dataset_name)
        
        logger.info(f"Loaded {len(self.data)} items from dataset")
    
    def __len__(self) -> int:
        """Get number of items in dataset."""
        return len(self.data)
    
    def __iter__(self):
        """Make the loader iterable."""
        self.current_index = 0
        return self
    
    def __next__(self):
        """Get next item in iteration."""
        if self.current_index >= len(self.data):
            raise StopIteration
        
        item = self.get_item(self.current_index)
        self.current_index += 1
        return item
    
    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Get a specific item from the dataset.
        
        Args:
            index: Index of the item to retrieve
            
        Returns:
            Dictionary containing table_id, table_name, table, question, answer, etc.
        """
        if index >= len(self.data):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.data)}")
        
        item = self.data[index]
        
        try:
            # Load table
            table = self.loader.get_table(item)
            
            # Extract common fields
            result = {
                'table_id': item.get('table_id', f"{self.dataset_name}_{index}"),
                'table_name': item.get('table_name', f"table_{index}"),
                'table': table,
                'question': item['question'],
                'answer': item.get('target_value', item.get('answer')),
                'paragraphs': item.get('paragraphs'),
                'index': index
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading item {index}: {e}")
            raise
    
    def get_batch(self, start_index: int, batch_size: int) -> List[Dict[str, Any]]:
        """Get a batch of items."""
        end_index = min(start_index + batch_size, len(self.data))
        return [self.get_item(i) for i in range(start_index, end_index)]
    
    def sample(self, n: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get a random sample of n items."""
        import random
        
        if seed is not None:
            random.seed(seed)
        
        if n >= len(self.data):
            return [self.get_item(i) for i in range(len(self.data))]
        
        indices = random.sample(range(len(self.data)), n)
        return [self.get_item(i) for i in indices]
