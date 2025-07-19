"""Single table question answering implementation."""

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
from ..prompts import load_prompt, configure_prompt_loader


logger = get_logger("core.weaver")


class TableQA(BaseQA):
    """Single table question answering system."""
    
    def __init__(self, config: Optional[WeaverConfig] = None):
        """Initialize TableQA with configuration."""
        if config is None:
            config = WeaverConfig.from_env()
        
        super().__init__(config)
        
        # Configure prompt loader with external prompts directory
        configure_prompt_loader(self.config.prompts_dir)
        
        # Data loader will be initialized as needed
        self.data_loader = None
        logger.info("TableQA initialized successfully")

    
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
    
    def ask(self, question_obj: Union[str, Dict[str, Any]], include_token_stats: bool = False, **kwargs) -> QAResult:
        """
        Answer a single question.
        
        Args:
            question_obj: Either a question string or a JSON object with question details
            include_token_stats: Whether to include token usage statistics in result
            **kwargs: Additional parameters (table_path, table_name, etc.)
            
        Returns:
            QAResult with answer and metadata
        """
        if isinstance(question_obj, str):
            # Simple string question - need table_path or table data
            question = question_obj
            table_path = kwargs.get('table_path')
            table_name = kwargs.get('table_name', 'table')
            
            if table_path:
                table = pd.read_csv(table_path)
            elif 'table' in kwargs:
                table = kwargs['table']
            else:
                raise ValueError("For string questions, provide table_path or table in kwargs")
            
            # Create a JSON object format
            question_obj = {
                'question': question,
                'table_name': table_name,
                'table': table,
                'paragraphs': kwargs.get('paragraphs'),
                'column_description_file': kwargs.get('column_description_file'),
                'table_schema_file': kwargs.get('table_schema_file'),
                'target_value': kwargs.get('target_value')  # for evaluation
            }
        
        return self._process_question(question_obj, include_token_stats, kwargs.get('dataset', 'default'))
    
    def _process_question(self, question_obj: Dict[str, Any], include_token_stats: bool = False, dataset: str = "default") -> QAResult:
        """Process a single question object."""
        start_time = time.time()
        
        # Extract required fields
        question = question_obj['question']
        table_id = question_obj.get('table_id', 'unknown')
        table_name = question_obj.get('table_name', 'table')
        
        logger.info(f"Processing question: {question[:100]}...")
        logger.info(f"Table: {table_name} (ID: {table_id})")
        
        try:
            # Step 1: Load table data
            logger.info("Loading table data...")
            table = self._load_table(question_obj)
            logger.info(f"Table loaded successfully: {table.shape[0]} rows, {table.shape[1]} columns")
            
            # Step 2: Load optional context
            logger.info("Loading optional context (paragraphs, descriptions, schema)...")
            paragraphs = self._load_paragraphs(question_obj)
            column_descriptions = self._load_column_descriptions(question_obj) 
            table_schema = self._load_table_schema(question_obj)
            
            context_info = []
            if paragraphs:
                context_info.append(f"paragraphs ({len(paragraphs)} chars)")
            if column_descriptions:
                context_info.append("column descriptions")
            if table_schema:
                context_info.append("table schema")
            
            if context_info:
                logger.info(f"Context loaded: {', '.join(context_info)}")
            else:
                logger.info("No additional context provided")
            
            # Step 3: Preprocess table
            logger.info("Preprocessing table for SQL compatibility...")
            preprocessor = TablePreprocessor(
                max_column_width=self.config.max_table_size,
                max_rows=self.config.max_table_size
            )
            
            clean_table_name, clean_table = preprocessor.clean_table(table, table_name)
            logger.info(f"Table preprocessed: '{table_name}' → '{clean_table_name}'")

            # Step 4: Generate column descriptions if not provided
            if column_descriptions is None:
                logger.info("Generating column descriptions using LLM...")
                column_descriptions = self._generate_column_descriptions(clean_table, clean_table_name, question)
                logger.info("Column descriptions generated")
            else:
                logger.info("Using provided column descriptions")

            # Step 5: Filter relevant columns (if enabled)
            if self.config.filter_relevant_columns:
                logger.info("Filtering relevant columns using LLM...")
                original_cols = len(clean_table.columns)
                clean_table = preprocessor.filter_relevant_columns(
                    clean_table, question, column_descriptions, self.llm, paragraphs, clean_table_name
                )
                logger.info(f"Column filtering complete: {original_cols} → {len(clean_table.columns)} columns")
            else:
                logger.info("Column filtering disabled, using all columns")

            # Step 6: Upload table to database
            logger.info("Uploading table to database...")
            self.database.upload_table(clean_table_name, clean_table)
            logger.info(f"Table uploaded: {clean_table_name} ({len(clean_table)} rows, {len(clean_table.columns)} columns)")
            
            # Step 7: Extract relevant paragraphs if provided
            if paragraphs:
                logger.info("Extracting relevant information from paragraphs...")
                relevant_paragraphs = self._get_relevant_paragraphs(paragraphs, clean_table, question)
                logger.info("Relevant paragraphs extracted")
            else:
                relevant_paragraphs = "No additional information provided."
                logger.info("No paragraphs to process")
            
            # Step 8: Create execution plan
            logger.info("Creating execution plan using LLM...")
            plan = self._create_plan(clean_table, clean_table_name, question, column_descriptions, relevant_paragraphs, dataset)
            logger.info("Execution plan created")
            
            # Step 9: Verify plan
            logger.info("Verifying and improving plan...")
            verified_plan = self._verify_plan(plan, clean_table, clean_table_name, question, column_descriptions, relevant_paragraphs, dataset)
            logger.info("Plan verified and improved")

            # Step 10: Generate executable code
            logger.info("Generating executable code from plan...")
            code = self._generate_code(verified_plan, clean_table, clean_table_name, question, column_descriptions, relevant_paragraphs, dataset)
            logger.info("Code generated successfully")

            # Step 11: Execute the code
            logger.info("Executing generated code...")
            final_table = self._execute_code(code, clean_table_name, question, relevant_paragraphs)
            logger.info(f"Code execution complete: final table shape {final_table.shape}")
            logger.info("Final table preview:")
            logger.info(final_table.head())

            # Step 12: Extract final answer
            logger.info("Extracting final answer from result table...")
            answer = self._extract_answer(final_table, question, relevant_paragraphs, dataset)
            logger.info("Answer extracted successfully")

            execution_time = time.time() - start_time
            logger.info(f"Question processed in {execution_time:.2f}s")
            
            # Step 13: Format answer and check correctness if gold answer provided
            is_correct = None
            gold_answer = question_obj.get('target_value')
            if gold_answer is not None:
                logger.info("Formatting answer and checking correctness...")
                # Use proper answer formatting from old weaver.py logic
                formatted_answer, is_correct = self._compare_answers(
                    final_table, question, gold_answer, relevant_paragraphs, dataset
                )
                answer = formatted_answer  # Use the formatted answer
                logger.info(f"Answer formatted and checked: {'✓ Correct' if is_correct else '✗ Incorrect'}")
            else:
                logger.info("No gold answer provided, skipping correctness check")

            # Step 14: Get token statistics if requested
            token_stats = None
            if include_token_stats:
                logger.info("Collecting token usage statistics...")
                token_stats = self.llm.get_usage_stats()
                logger.info("Token statistics collected")

            logger.info(f"Question processing complete! Answer: {answer}")

            return QAResult(
                question=question,
                answer=answer,
                plan=verified_plan,
                sql_code=code,
                is_correct=is_correct,
                gold_answer=gold_answer,
                table_id=table_id,
                token_stats=token_stats
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error processing question after {execution_time:.2f}s: {e}")

            # Get token statistics if requested (even for errors)
            token_stats = None
            if include_token_stats:
                logger.info("Collecting token statistics for error case...")
                token_stats = self.llm.get_usage_stats()
                
            return QAResult(
                question=question,
                answer=f"Error: {str(e)}",
                table_id=table_id,
                token_stats=token_stats
            )
    
    def _load_table(self, question_obj: Dict[str, Any]) -> pd.DataFrame:
        """Load table from question object."""
        # If table is directly provided
        if 'table' in question_obj:
            return question_obj['table']
        
        # If table_file_name is provided (dataset format)
        if 'table_file_name' in question_obj:
            table_path = question_obj['table_file_name']
            
            # Handle absolute vs relative paths
            if not os.path.isabs(table_path):
                # Try in datasets directory
                
                dataset_path = self.config.datasets_dir / table_path
                if dataset_path.exists():
                    table_path = str(dataset_path)
            
            if table_path.endswith('.csv'):
                return pd.read_csv(table_path)
            elif table_path.endswith('.json'):
                with open(table_path, 'r') as f:
                    table_data = json.load(f)
                return pd.DataFrame(table_data)
            else:
                raise ValueError(f"Unsupported table format: {table_path}")
        
        raise ValueError("No table data found in question object")
    
    def _load_paragraphs(self, question_obj: Dict[str, Any]) -> Optional[str]:
        """Load additional paragraphs if provided."""
        paragraphs = question_obj.get('paragraphs')
        if paragraphs and isinstance(paragraphs, str) and len(paragraphs.strip()) > 0:
            return paragraphs.strip()
        return None
    
    def _load_column_descriptions(self, question_obj: Dict[str, Any]) -> Optional[str]:
        """Load column descriptions from file if provided."""
        desc_file = question_obj.get('column_description_file')
        if desc_file and os.path.exists(desc_file):
            try:
                with open(desc_file, 'r') as f:
                    return f.read().strip()
            except Exception as e:
                logger.warning(f"Failed to load column descriptions from {desc_file}: {e}")
        return None
    
    def _load_table_schema(self, question_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Load table schema from file if provided."""
        schema_file = question_obj.get('table_schema_file')
        if schema_file and os.path.exists(schema_file):
            try:
                with open(schema_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load table schema from {schema_file}: {e}")
        return None
    
    def _generate_column_descriptions(self, table: pd.DataFrame, table_name: str, question: str) -> str:
        """Generate column descriptions using LLM."""
        prompt = f"""
        Give me the column name, data type, formatting needed in detail, and column descriptions in detail, for the context of question on the table.
        Also, give a small description of the table using table name and table data given.
        
        Table name: {table_name}
        Table columns: {list(table.columns)}
        Table preview:
        {table.head().to_html()}
        
        Question: {question}
        
        Provide detailed descriptions for each column and the overall table.
        """
        
        return self.llm.call(prompt)
    
    def _get_relevant_paragraphs(self, paragraphs: str, table: pd.DataFrame, question: str) -> str:
        """Extract relevant information from paragraphs using LLM."""
        if not paragraphs:
            return "No additional information provided."
        
        prompt = f"""
        Given the question, some paragraphs and the table, you need to extract the useful information in the paragraphs to answer the question.
        You can use the table columns context as well to extract the relevant information from the paragraphs.
        
        Paragraphs: {paragraphs}
        Table columns: {list(table.columns)}
        Question: {question}
        
        Extract and return only the relevant information from the paragraphs.
        """
        
        return self.llm.call(prompt)
    
    def _create_plan(self, table: pd.DataFrame, table_name: str, question: str, 
                    column_descriptions: str, relevant_paragraphs: str, dataset: str = "default") -> str:
        """Create execution plan using LLM."""
        # Load prompts using the new system (no file I/O for built-ins!)
        plan_prompt = load_prompt("planner_prompt", dataset)
        few_shot = load_prompt("few_shot_plan", dataset)
        
        prompt = plan_prompt + "\n\n" + few_shot + f"""

        Solve for this:
        Table name: {table_name}
        {table.to_html()}
        Column descriptions: {column_descriptions}
        Paragraphs: {relevant_paragraphs}
        Question: {question}
        
        Only give the step by step plan and remove any extra explanation or Code.
        Output format:
        Step 1: SQL - [Instruction that can be used to write MySQL query]
        Step 2: Either SQL or LLM
        Step 3: ...
        
        Plan:
        """
        
        return self.llm.call(prompt)
    
    def _verify_plan(self, plan: str, table: pd.DataFrame, table_name: str, 
                    question: str, column_descriptions: str, relevant_paragraphs: str, dataset: str = "default") -> str:
        """Verify and improve the plan using LLM."""
        # Load prompts using the new system (no file I/O for built-ins!)
        base_verify_prompt = load_prompt("verify_plan", dataset)
        
        # Construct full prompt with context
        verify_prompt = base_verify_prompt + f"""
        
        Table name: {table_name}
        Table: {table.to_html()}
        Column descriptions: {column_descriptions}
        Paragraphs: {relevant_paragraphs}
        Question: {question}
        
        Old Plan:
        {plan}
        """
        
        return self.llm.call(verify_prompt)
    
    def _generate_code(self, plan: str, table: pd.DataFrame, table_name: str,
                      question: str, column_descriptions: str, relevant_paragraphs: str, dataset: str = "default") -> str:
        """Generate executable code from plan."""
        # Load prompts using the new system (no file I/O for built-ins!)
        execute_prompt = load_prompt("execute_prompt", dataset)
        
        prompt = execute_prompt + f"""
        
        Table name: {table_name}
        Paragraphs: {relevant_paragraphs}
        Schema: {list(table.columns)}
        Column Descriptions: {column_descriptions}
        
        Table: (This is a Sample table and the actual table can have more rows than below provided)
        {table.to_html()}
        
        Question: {question}
        Plan: {plan}
        
        Give me code for solving the question, and no other explanations. 
        Keep in mind the column data formats while writing SQL code.
        """
        
        return self.llm.call(prompt)
    
    def _execute_code(self, code: str, table_name: str, question: str, relevant_paragraphs: str) -> pd.DataFrame:
        """Execute the generated code and return final table."""
        df = self.database.get_table_data(table_name)
        tmp_df = df
        current_table_name = table_name
        
        # Split the code by steps and execute each one
        steps = re.split(r"Step \d+", code)
        logger.info('-----------------EXECUTING CODE------------------')
        
        for num, step in enumerate(steps):
            if not step.strip():
                continue
                
            try:
                if 'SQL' in step[:20] or 'sql' in step[:20]:
                    # Extract and execute SQL query
                    sql_pattern = r"\b(?:CREATE TABLE|SELECT)\b.*?;"
                    matches = re.findall(sql_pattern, step, re.DOTALL)
                    
                    for match in matches:
                        logger.info('--------------------SQL STEP--------------------------')
                        logger.info(match)
                        
                        result = self.database.execute_query(match)
                        
                        if result.success:
                            if result.data is not None:
                                # SELECT query returned data
                                tmp_df = result.data
                                logger.info(f"SQL returned {len(tmp_df)} rows")
                            elif result.table_name:
                                # CREATE TABLE query created new table
                                current_table_name = result.table_name
                                tmp_df = self.database.get_table_data(current_table_name)
                                logger.info(f"Created table: {current_table_name}")
                        else:
                            # SQL execution failed - suppress error log for cleaner output
                            logger.debug(f"SQL execution failed: {result.error}")
                            continue
                        
                        logger.debug(f"Current table:\n{tmp_df.to_string()}")

                elif 'LLM' in step[:20] or 'llm' in step[:20]:
                    logger.info('-----------------------LLM STEP---------------------------')
                    logger.info(step)
                    
                    # Extract information from LLM step
                    cols = self._get_prev_colname(step)
                    final_cols = []
                    for col in cols:
                        if col in tmp_df.columns:
                            final_cols.append(col)
                        else:
                            logger.warning(f"Column '{col}' not found in the DataFrame.")
                    
                    if not final_cols:
                        logger.warning("No valid columns found for LLM step, skipping")
                        continue

                    step_table_name = self._get_tablename(step)
                    if step_table_name:
                        current_table_name = step_table_name
                        tmp_df = self.database.get_table_data(current_table_name)
                    
                    step_prompt = self._get_new_prompt(step)
                    new_col_name = self._get_new_colname(step)
                    
                    if not step_prompt or not new_col_name:
                        logger.warning("Missing prompt or column name for LLM step, skipping")
                        continue

                    # Process in batches to handle large tables
                    batch_size = 10
                    new_col = []

                    for start in range(0, len(tmp_df), batch_size):
                        end = start + batch_size
                        batch_df = tmp_df.iloc[start:end]
                        batch_column_value = batch_df[final_cols]

                        # Create LLM prompt for this batch
                        llm_prompt = self._create_llmstep_prompt(
                            step_prompt, batch_column_value, relevant_paragraphs, question
                        )
                        
                        batch_col = self.llm.call(llm_prompt)
                        
                        # Split the response by '#' to get individual values
                        batch_values = batch_col.split('#')
                        
                        # Clean up any empty values
                        batch_values = [val.strip() for val in batch_values if val.strip()]
                        
                        # Handle length mismatch
                        if len(batch_values) != len(batch_column_value):
                            logger.warning(f"BATCH LENGTH MISMATCH: Expected {len(batch_column_value)}, got {len(batch_values)}")
                            
                            # Try to fix by padding or truncating
                            if len(batch_values) < len(batch_column_value):
                                # Pad with the last value or empty string
                                while len(batch_values) < len(batch_column_value):
                                    batch_values.append(batch_values[-1] if batch_values else "")
                            elif len(batch_values) > len(batch_column_value):
                                # Truncate to expected length
                                batch_values = batch_values[:len(batch_column_value)]
                        
                        new_col.extend(batch_values)

                    logger.info(f'Final new column values: {new_col}')
                    logger.info(f'Final column length: {len(new_col)}, Expected: {len(tmp_df)}')
                    
                    # Add new column if lengths match
                    if len(new_col) == len(tmp_df) and new_col_name:
                        tmp_df[new_col_name] = new_col
                        # Upload updated table back to database
                        self.database.upload_table(current_table_name, tmp_df)
                        logger.info(f'LLM updated table: {current_table_name}, with column: {new_col_name}')
                    else:
                        logger.error(f'LLM column length mismatch: expected {len(tmp_df)}, got {len(new_col)}')
                        logger.error(f"This will cause the step to fail!")
                    
                    logger.debug(f"Updated table:\n{tmp_df.to_string()}")
                    
                else:
                    logger.debug(f"Unrecognized step type in step {num}: {step[:50]}")

                # Check if table became empty
                if tmp_df.shape[0] == 0:
                    logger.warning("Table became empty after step, returning original table")
                    return df
                
                df = tmp_df

            except Exception as e:
                logger.error(f'Error in step {num}: {e}')
                continue

        logger.info(f'Final table shape: {df.shape}')
        return df
    
    def _extract_answer(self, final_table: pd.DataFrame, question: str, 
                       relevant_paragraphs: str, dataset: str) -> str:
        """Extract final answer from the result table."""
        # Load prompts using the new system (no file I/O for built-ins!)
        prompt = load_prompt("extract_answer", dataset)
        
        prompt += f"""
        
        Table: {final_table.to_html(index=False)}
        Paragraphs: {relevant_paragraphs}
        Question: {question}
        
        Answer:
        """
        
        return self.llm.call(prompt)

    def _format_answer(self, final_table: pd.DataFrame, question: str, gold_answer: str,
                      relevant_paragraphs: str, dataset: str) -> str:
        """
        Format answer using dataset-specific prompts, similar to old weaver.py format_answer method.
        
        Args:
            final_table: Result table from code execution
            question: Original question
            gold_answer: Expected answer
            relevant_paragraphs: Context paragraphs
            dataset: Dataset name for prompt selection
            
        Returns:
            Formatted answer string
        """
        logger.info('-----------------FORMATTING ANSWER------------------')
        logger.debug(f'Relevant paragraphs: {relevant_paragraphs}')
        
        # First, extract answer from table using the existing method
        answer = self._extract_answer(final_table, question, relevant_paragraphs, dataset)
        logger.info(f"Initial extracted answer: {answer}")

        # Second step: Format the answer using dataset-specific formatting
        answer_formatting_prompt = load_prompt("format_answer", dataset)
        
        answer_formatting_prompt += f'''
        Solve for this-
        Answer: {answer}
        Gold Answer: {gold_answer}
        Your Output:
        '''
        
        # Get formatted answer
        formatted_answer = self.llm.call(answer_formatting_prompt)
        logger.info(f"Formatted answer: {formatted_answer}")
        return formatted_answer
    
    def _compare_answers(self, final_table: pd.DataFrame, question: str, gold_answer: str,
                        relevant_paragraphs: str, dataset: str) -> tuple:
        """
        Compare model answer with gold answer using proper formatting, similar to old weaver.py compare method.
        
        Args:
            final_table: Result table from code execution
            question: Original question  
            gold_answer: Expected answer
            relevant_paragraphs: Context paragraphs
            dataset: Dataset name
            
        Returns:
            Tuple of (formatted_answer, is_correct)
        """
        # Format the answer using dataset-specific prompts
        formatted_answer = self._format_answer(final_table, question, gold_answer, relevant_paragraphs, dataset)
        
        logger.info(f'Gold answer: {gold_answer}')
        logger.info(f'Model answer: {formatted_answer}')

        # Check if answers match
        if formatted_answer.strip() == str(gold_answer).strip():
            logger.info(f'Model answer: {formatted_answer} and Gold answer {gold_answer} match')
            is_correct = True
        else:
            logger.info(f'Model answer: {formatted_answer} and Gold answer {gold_answer} do not match')
            is_correct = False
        
        return formatted_answer, is_correct
    
    # Helper methods for code execution (ported from old weaver.py)
    def _get_new_colname(self, step: str) -> Optional[str]:
        """Extract new column name from LLM step."""
        match = re.search(r"(?<=column name: )(\S+)", step, re.IGNORECASE)
        if match:
            new_column_name = match.group(1)
            new_column_name = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', new_column_name)
            logger.debug(f"New column name: {new_column_name}")
            return new_column_name
        else:
            logger.debug("New column name not found.")
            return None
    
    def _get_tablename(self, step: str) -> Optional[str]:
        """Extract table name from step."""
        match = re.search(r"(?<=table name: )(\S+)", step, re.IGNORECASE)
        if match:
            name = match.group(1)
            name = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', name)
            logger.debug(f"Table name: {name}")
            return name
        else:
            logger.debug("Table name not found.")
            return None
    
    def _get_new_prompt(self, step: str) -> Optional[str]:
        """Extract LLM prompt from step."""
        match = re.search(r'(?<=LLM prompt: )(.*)', step, re.IGNORECASE)
        if match:
            llm_prompt = match.group(1)
            logger.debug(f"LLM Prompt: {llm_prompt}")
            return llm_prompt
        else:
            logger.debug("LLM prompt not found.")
            return None
    
    def _get_prev_colname(self, step: str) -> List[str]:
        """Extract previous column names from step."""
        match = re.search(r"(?<=to be used: )(.*)", step, re.IGNORECASE)
        if match:
            column_names = match.group(1)
            column_names = column_names.split(',')
            column_names = [col.strip() for col in column_names]
            return column_names
        else:
            logger.debug("Previous column names not found.")
            return []
    
    def _create_llmstep_prompt(self, llm_step: str, column_value: pd.DataFrame, 
                              paragraphs: str, question: str) -> str:
        """Create prompt for LLM step processing."""
        input_count = len(column_value)
        
        prompt = f"""
        Given a column and step you need to perform on it with some paragraphs which can be useful-
        
        Column: {column_value.to_string()}
        Step to solve the question: {llm_step}
        Question: {question}
        Paragraphs: {paragraphs}
        
        CRITICAL INSTRUCTIONS: 
        - You must provide EXACTLY {input_count} values in your response
        - Separate values by '#' character only
        - Do not provide any explanation or additional text
        - Return only a list (separate values by '#') that can be added to a dataframe as a new column
        - Any value should not be more than 3 words (or each value should be as short as possible)
        - Size of output column MUST be same as input column: {input_count} values
        - Example format: value1#value2#value3 (for 3 input rows)
        
        Your response (exactly {input_count} values separated by #):
        """
        return prompt
    
    def evaluate_dataset(self, dataset_name: str, data_path: str, 
                        num_samples: Optional[int] = None, 
                        start_index: int = 0,
                        include_token_stats: bool = False) -> List[QAResult]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataset_name: Name of the dataset (wikitq, tabfact, etc.)
            data_path: Path to the JSON file containing questions
            num_samples: Number of samples to process (None for all)
            start_index: Starting index for processing
            
        Returns:
            List of QAResult objects
        """
        logger.info(f"Starting evaluation on {dataset_name} dataset: {data_path}")
        
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
            
            logger.info(f"Processing sample {i+1}/{end_index}")
            result = self._process_question(question_obj, include_token_stats, dataset_name)
            results.append(result)
            
            # Log intermediate results
            if result.is_correct is not None:
                correct_count = sum(1 for r in results if r.is_correct)
                accuracy = correct_count / len(results)
                logger.info(f"Current accuracy: {accuracy:.2%} ({correct_count}/{len(results)})")
        
        # Save results
        results_file = self.config.results_dir / f"{self.config.llm.model.replace('/', '_')}_{dataset_name}_results.json"
        results_data = [r.to_dict() for r in results]
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        return results
