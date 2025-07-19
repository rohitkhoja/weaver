"""Multi-table question answering implementation."""

import os
import re
import time
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from .base import BaseQA, QAResult
from ..config.settings import WeaverConfig
from ..config.logging_config import get_logger
from ..database.manager import DatabaseManager
from ..llm.client import LLMClient
from ..data.loader import DataLoader
from ..data.preprocessor import TablePreprocessor


logger = get_logger("core.weaver_multi")


class MultiTableQA(BaseQA):
    """Multi-table question answering system."""
    
    def __init__(self, config: Optional[WeaverConfig] = None):
        """Initialize MultiTableQA with configuration."""
        if config is None:
            config = WeaverConfig.from_env()
        
        super().__init__(config)
        
        # Initialize data loader
        self.data_loader = DataLoader()
        
        # Track loaded tables
        self.loaded_tables = {}
        
        logger.info("MultiTableQA initialized successfully")
    

    
    def _setup_database(self) -> None:
        """Setup database connection."""
        connection_string = self.config.database.get_connection_string()
        db_type = self.config.database.db_type
        self.database = DatabaseManager(connection_string, db_type)
        logger.info(f"Database initialized: {db_type} - {connection_string}")
    
    def _setup_llm(self) -> None:
        """Setup LLM client."""
        self.llm = LLMClient(self.config.llm)
        logger.info(f"LLM client initialized: {self.config.llm.model}")
    
    def load_tables(self, tables: List[Dict[str, Any]]) -> None:
        """
        Load multiple tables into the system.
        
        Args:
            tables: List of table definitions, each containing:
                - name: Table name/alias
                - path: Path to table file (CSV/JSON)
                - Optional: description, column_descriptions, schema
        """
        logger.info(f"Loading {len(tables)} tables")
        
        preprocessor = TablePreprocessor(
            max_column_width=self.config.max_table_size,
            max_rows=self.config.max_table_size
        )
        
        for table_def in tables:
            table_name = table_def['name']
            table_path = table_def['path']
            
            logger.info(f"Loading table: {table_name} from {table_path}")
            
            # Load table data
            if table_path.endswith('.csv'):
                table = pd.read_csv(table_path)
            elif table_path.endswith('.json'):
                with open(table_path, 'r') as f:
                    table_data = json.load(f)
                table = pd.DataFrame(table_data)
            else:
                raise ValueError(f"Unsupported table format: {table_path}")
            
            # Clean table
            clean_table_name, clean_table = preprocessor.clean_table(table, table_name)
            
            # Upload to database
            self.database.upload_table(clean_table_name, clean_table)
            
            # Store table metadata
            self.loaded_tables[table_name] = {
                'clean_name': clean_table_name,
                'data': clean_table,
                'original_path': table_path,
                'description': table_def.get('description', ''),
                'column_descriptions': table_def.get('column_descriptions', ''),
                'schema': table_def.get('schema', {})
            }
            
            logger.info(f"Loaded table {table_name} with {len(clean_table)} rows, {len(clean_table.columns)} columns")
    
    def ask(self, question: str, **kwargs) -> QAResult:
        """
        Answer a question using loaded tables.
        
        Args:
            question: Question string
            **kwargs: Additional parameters
            
        Returns:
            QAResult with answer and metadata
        """
        if not self.loaded_tables:
            raise ValueError("No tables loaded. Use load_tables() first.")
        
        question_obj = {
            'question': question,
            'tables': self.loaded_tables,
            'target_value': kwargs.get('target_value')  # for evaluation
        }
        
        return self._process_question(question_obj)
    
    def _process_question(self, question_obj: Dict[str, Any]) -> QAResult:
        """Process a multi-table question."""
        start_time = time.time()
        
        question = question_obj['question']
        tables = question_obj.get('tables', self.loaded_tables)
        
        logger.info(f"Processing multi-table question: {question[:100]}...")
        logger.info(f"Available tables: {list(tables.keys())}")
        
        try:
            # Identify relevant tables for this question
            relevant_tables = self._identify_relevant_tables(question, tables)
            logger.info(f"Relevant tables: {list(relevant_tables.keys())}")
            
            # Generate multi-table column descriptions
            table_descriptions = self._generate_multi_table_descriptions(relevant_tables, question)
            
            # Create multi-table execution plan
            plan = self._create_multi_table_plan(relevant_tables, question, table_descriptions)
            
            # Verify plan
            verified_plan = self._verify_multi_table_plan(plan, relevant_tables, question, table_descriptions)
            
            # Generate and execute multi-table code
            code = self._generate_multi_table_code(verified_plan, relevant_tables, question, table_descriptions)
            
            # Execute the code and get final result
            final_table = self._execute_multi_table_code(code, relevant_tables)
            
            # Extract final answer
            answer = self._extract_multi_table_answer(final_table, question, relevant_tables)
            
            # Calculate confidence
            confidence = self._calculate_confidence(final_table, answer)
            
            execution_time = time.time() - start_time
            logger.info(f"Multi-table question processed in {execution_time:.2f}s")
            
            # Check correctness if gold answer provided
            is_correct = None
            gold_answer = question_obj.get('target_value')
            if gold_answer is not None:
                is_correct = self._check_correctness(answer, gold_answer)
            
            return QAResult(
                question=question,
                answer=answer,
                confidence=confidence,
                plan=verified_plan,
                sql_code=code,
                is_correct=is_correct,
                gold_answer=gold_answer
            )
            
        except Exception as e:
            logger.error(f"Error processing multi-table question: {e}")
            return QAResult(
                question=question,
                answer=f"Error: {str(e)}",
                confidence=0.0
            )
    
    def _identify_relevant_tables(self, question: str, tables: Dict[str, Any]) -> Dict[str, Any]:
        """Identify which tables are relevant for answering the question."""
        # For now, use all loaded tables
        # This could be enhanced with semantic similarity or keyword matching
        
        question_lower = question.lower()
        relevant_tables = {}
        
        for table_name, table_info in tables.items():
            # Simple heuristic: check if table name appears in question
            if table_name.lower() in question_lower:
                relevant_tables[table_name] = table_info
                continue
            
            # Check if any column names appear in question
            table_data = table_info['data']
            for col in table_data.columns:
                if col.lower() in question_lower:
                    relevant_tables[table_name] = table_info
                    break
        
        # If no tables identified as relevant, use all tables
        if not relevant_tables:
            relevant_tables = tables
        
        return relevant_tables
    
    def _generate_multi_table_descriptions(self, tables: Dict[str, Any], question: str) -> str:
        """Generate descriptions for multiple tables."""
        descriptions = []
        
        for table_name, table_info in tables.items():
            table_data = table_info['data']
            
            # Use existing description if available
            if table_info.get('column_descriptions'):
                descriptions.append(f"Table: {table_name}\n{table_info['column_descriptions']}")
            else:
                # Generate description using LLM
                prompt = f"""
                Describe this table in the context of the question.
                
                Table name: {table_name}
                Table columns: {list(table_data.columns)}
                Table preview:
                {table_data.head().to_html()}
                
                Question: {question}
                
                Provide a brief description of the table and its columns.
                """
                
                description = self.llm.call(prompt)
                descriptions.append(f"Table: {table_name}\n{description}")
        
        return "\n\n".join(descriptions)
    
    def _create_multi_table_plan(self, tables: Dict[str, Any], question: str, descriptions: str) -> str:
        """Create execution plan for multi-table query."""
        table_info = []
        for table_name, table_data in tables.items():
            clean_name = table_data['clean_name']
            data = table_data['data']
            table_info.append(f"Table: {table_name} (SQL name: {clean_name})")
            table_info.append(f"Columns: {list(data.columns)}")
            table_info.append(f"Sample data:\n{data.head(3).to_html()}")
            table_info.append("")
        
        table_context = "\n".join(table_info)
        
        prompt = f"""
        You are a data scientist expert in SQL and multi-table analysis. Create a step-by-step plan to answer the question using the provided tables.
        
        Available tables:
        {table_context}
        
        Table descriptions:
        {descriptions}
        
        Question: {question}
        
        Create a plan that may involve:
        - Joining tables where appropriate
        - Filtering and aggregating data
        - Using SQL for data manipulation
        - LLM steps for natural language processing if needed
        
        Output format:
        Step 1: SQL - [Instruction for SQL operation across tables]
        Step 2: SQL/LLM - [Next operation]
        Step 3: ...
        
        Plan:
        """
        
        return self.llm.call(prompt)
    
    def _verify_multi_table_plan(self, plan: str, tables: Dict[str, Any], question: str, descriptions: str) -> str:
        """Verify the multi-table execution plan."""
        table_info = []
        for table_name, table_data in tables.items():
            clean_name = table_data['clean_name']
            data = table_data['data']
            table_info.append(f"Table: {table_name} (SQL: {clean_name}, Columns: {list(data.columns)})")
        
        table_context = "\n".join(table_info)
        
        verify_prompt = f"""
        Verify if this plan will correctly answer the multi-table question.
        
        Available tables: {table_context}
        Question: {question}
        Current plan: {plan}
        
        Check:
        1. Are the correct tables being used?
        2. Are joins properly specified?
        3. Will the plan produce the correct answer?
        
        If the plan is correct, return it as-is.
        If not, provide an improved plan.
        
        Plan:
        """
        
        return self.llm.call(verify_prompt)
    
    def _generate_multi_table_code(self, plan: str, tables: Dict[str, Any], question: str, descriptions: str) -> str:
        """Generate executable SQL code for multi-table operations."""
        table_info = []
        for table_name, table_data in tables.items():
            clean_name = table_data['clean_name']
            data = table_data['data']
            table_info.append(f"Table: {table_name} -> SQL table name: {clean_name}")
            table_info.append(f"Schema: {list(data.columns)}")
            table_info.append(f"Sample:\n{data.head(2).to_html()}")
            table_info.append("")
        
        table_context = "\n".join(table_info)
        
        prompt = f"""
        Generate executable MySQ code based on the plan.

        Available tables:
        {table_context}
        
        Table descriptions: {descriptions}
        Question: {question}
        Plan: {plan}
        
        Generate valid SQL code that:
        1. Uses the correct table names (SQL names provided above)
        2. Includes proper joins where needed
        3. Handles data types correctly
        4. Produces the final result
        
        Code:
        """
        
        return self.llm.call(prompt)
    
    def _execute_multi_table_code(self, code: str, tables: Dict[str, Any]) -> pd.DataFrame:
        """Execute multi-table SQL code."""
        # Split the code by steps and execute each one
        steps = re.split(r"Step \d+", code)
        
        # Start with an empty result
        result_df = pd.DataFrame()
        
        for step in steps:
            if not step.strip():
                continue
                
            try:
                if 'SQL' in step[:20] or 'sql' in step[:20]:
                    # Extract and execute SQL query
                    sql_pattern = r"\b(?:CREATE TABLE|SELECT|WITH)\b.*?;"
                    matches = re.findall(sql_pattern, step, re.DOTALL | re.IGNORECASE)
                    
                    for match in matches:
                        logger.debug(f"Executing multi-table SQL: {match}")
                        result = self.database.execute_query(match)
                        
                        if result.success and result.data is not None:
                            result_df = result.data
                        elif result.table_name:
                            result_df = self.database.get_table_data(result.table_name)
                            
            except Exception as e:
                logger.error(f"Error executing multi-table step: {e}")
                continue
        
        # If no result from steps, try executing the entire code as one query
        if result_df.empty:
            try:
                # Look for SELECT statements in the code
                select_pattern = r"\bSELECT\b.*?(?=;|$)"
                select_matches = re.findall(select_pattern, code, re.DOTALL | re.IGNORECASE)
                
                if select_matches:
                    # Execute the last SELECT statement
                    last_select = select_matches[-1].strip()
                    if not last_select.endswith(';'):
                        last_select += ';'
                    
                    logger.debug(f"Executing final SELECT: {last_select}")
                    result = self.database.execute_query(last_select)
                    
                    if result.success and result.data is not None:
                        result_df = result.data
                        
            except Exception as e:
                logger.error(f"Error executing final query: {e}")
        
        return result_df
    
    def _extract_multi_table_answer(self, final_table: pd.DataFrame, question: str, tables: Dict[str, Any]) -> str:
        """Extract final answer from multi-table result."""
        table_context = "\n".join([f"- {name}: {info['description']}" for name, info in tables.items()])
        
        prompt = f"""
        Extract the final answer from this result table that answers the multi-table question.
        
        Original tables used: {table_context}
        Question: {question}
        Result table:
        {final_table.to_html(index=False)}
        
        Provide a concise, specific answer:
        """
        
        return self.llm.call(prompt)
    
    def _calculate_confidence(self, final_table: pd.DataFrame, answer: str) -> float:
        """Calculate confidence score for multi-table answer."""
        confidence = 0.4  # Lower base confidence for multi-table
        
        if len(final_table) == 1:
            confidence += 0.4  # Higher confidence for single result
        elif len(final_table) <= 3:
            confidence += 0.3  # Medium confidence for few results
        elif len(final_table) <= 10:
            confidence += 0.1  # Lower confidence for many results
        
        if answer and len(answer.strip()) > 0 and "error" not in answer.lower():
            confidence += 0.2  # Higher confidence for non-empty, non-error answers
        
        return min(confidence, 1.0)
    
    def _check_correctness(self, model_answer: str, gold_answer: str) -> bool:
        """Check if model answer matches gold answer."""
        return str(model_answer).strip().lower() == str(gold_answer).strip().lower()
    
    def evaluate_dataset(self, dataset_name: str, data_path: str, 
                        num_samples: Optional[int] = None, 
                        start_index: int = 0) -> List[QAResult]:
        """
        Evaluate model on a multi-table dataset.
        
        Args:
            dataset_name: Name of the dataset
            data_path: Path to the JSON file containing questions
            num_samples: Number of samples to process (None for all)
            start_index: Starting index for processing
            
        Returns:
            List of QAResult objects
        """
        logger.info(f"Starting multi-table evaluation on {dataset_name} dataset: {data_path}")
        
        # Load dataset
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Determine sample range
        end_index = len(data)
        if num_samples:
            end_index = min(start_index + num_samples, len(data))
        
        logger.info(f"Processing {end_index - start_index} samples from index {start_index} to {end_index-1}")
        
        results = []
        for i in range(start_index, end_index):
            question_obj = data[i]
            
            # For multi-table datasets, we might need to load tables for each question
            # This depends on the dataset format
            logger.info(f"Processing multi-table sample {i+1}/{end_index}")
            result = self._process_question(question_obj)
            results.append(result)
            
            # Log intermediate results
            if result.is_correct is not None:
                correct_count = sum(1 for r in results if r.is_correct)
                accuracy = correct_count / len(results)
                logger.info(f"Current accuracy: {accuracy:.2%} ({correct_count}/{len(results)})")
        
        # Save results
        results_file = self.config.results_dir / f"{self.config.llm.model.replace('/', '_')}_{dataset_name}_multi_results.json"
        results_data = [r.to_dict() for r in results]
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Multi-table results saved to: {results_file}")
        return results


# Legacy compatibility
HybridQAMulti = MultiTableQA
