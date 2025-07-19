"""Enhanced LLM client using LiteLLM for multi-provider support."""

from typing import List, Optional, Dict, Any
from litellm import completion, batch_completion

from ..config.logging_config import get_logger
from ..config.settings import LLMConfig


logger = get_logger("llm.client")


class LLMClient:
    """
    LLM client wrapper using LiteLLM for multi-provider support.
    
    LiteLLM automatically handles API keys from environment variables.
    Set the appropriate environment variable for your provider:
    
    Example usage:
    ```python
    import os
    from litellm import completion
    
    # Set your API key
    os.environ["OPENAI_API_KEY"] = "your-api-key"
    
    # Use model in format "provider/model" or configure provider separately
    response = completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    ```
    
    Common environment variables:
    - OPENAI_API_KEY for OpenAI models
    - ANTHROPIC_API_KEY for Claude models  
    - GEMINI_API_KEY for Google Gemini
    - AZURE_API_KEY for Azure OpenAI
    - And many more - see LiteLLM documentation
    """
    
    def __init__(self, config: LLMConfig):
        """Initialize LLM client with configuration."""
        self.config = config
        self.model = config.model
        self.count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        
        logger.info(f"Initialized LLM client with model: {self.model}")
        logger.info("Ensure your API key is set as an environment variable for your LLM provider")
    
    def __repr__(self) -> str:
        """String representation of the client."""
        return (f'LLMClient(model={self.model}, calls={self.count}, '
                f'input_tokens={self.total_input_tokens}, '
                f'output_tokens={self.total_output_tokens}, '
                f'total_tokens={self.total_tokens})')
    
    def call(self, prompt: str, system_message: Optional[str] = None) -> str:
        """
        Make a single LLM call.
        
        Args:
            prompt: User prompt
            system_message: Optional system message (defaults to data science expert)
            
        Returns:
            LLM response text
        """
        if system_message is None:
            system_message = "You are a data scientist expert in SQL and LLM."
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        try:
            logger.debug(f"Making API call #{self.count + 1}")

            response = completion(
                model=self.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            # Track token usage
            usage = response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
            
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_tokens += total_tokens
            self.count += 1
            
            logger.info(f"API Call #{self.count} - Input: {input_tokens}, "
                       f"Output: {output_tokens}, Total: {total_tokens} tokens")
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
    def call_batch(self, prompts: List[str], system_message: Optional[str] = None) -> List[str]:
        """
        Make batch LLM calls.
        
        Args:
            prompts: List of user prompts
            system_message: Optional system message
            
        Returns:
            List of LLM responses
        """
        if system_message is None:
            system_message = "You are a data scientist expert in SQL and LLM prompts."
        
        # Prepare messages for batch processing
        base_message = [{"role": "system", "content": system_message}]
        messages_batch = [
            base_message + [{"role": "user", "content": prompt}] 
            for prompt in prompts
        ]
        
        try:
            logger.debug(f"Making batch API call with {len(prompts)} prompts")
            
            responses = batch_completion(
                model=self.model,
                messages=messages_batch,
                temperature=self.config.temperature
            )
            
            # Track token usage for all responses
            total_batch_tokens = 0
            results = []
            
            for i, response in enumerate(responses):
                usage = response.usage
                input_tokens = usage.prompt_tokens
                output_tokens = usage.completion_tokens
                total_tokens = usage.total_tokens
                
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                self.total_tokens += total_tokens
                total_batch_tokens += total_tokens
                
                logger.info(f"Batch API Call #{self.count + i + 1} - Input: {input_tokens}, "
                           f"Output: {output_tokens}, Total: {total_tokens} tokens")
                
                results.append(response.choices[0].message.content.strip())
            
            self.count += len(prompts)
            logger.info(f"Batch completed: {len(prompts)} calls, {total_batch_tokens} total tokens")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch LLM API call failed: {e}")
            raise
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "model": self.model,
            "total_calls": self.count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "average_tokens_per_call": self.total_tokens / self.count if self.count > 0 else 0
        }
    
    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self.count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        logger.info("Usage statistics reset")


def create_llm_client(config: LLMConfig) -> LLMClient:
    """Factory function to create LLM client."""
    return LLMClient(config)
