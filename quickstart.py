"""
Basic usage examples for Weaver TableQA system.

This demonstrates the new API with support for:
1. Single question answering
2. Batch processing with JSON objects  
3. Optional context (paragraphs, column descriptions, schema)

Setup Instructions:
==================

1. Set your API key as an environment variable:

import os

# For OpenAI
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# For Anthropic Claude  
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key"

# For Google Gemini
os.environ["GEMINI_API_KEY"] = "your-gemini-api-key"

# Or set in your shell:
# export OPENAI_API_KEY=your-openai-api-key

2. Configure your model in LiteLLM format:
   - Use "openai/gpt-4o" or provider="openai", model="gpt-4o"
   - Use "anthropic/claude-3-sonnet-20240229" for Claude
   - Use "gemini/gemini-pro" for Gemini
   - See LiteLLM docs for all supported providers
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the Python path so we can import weaver
# This allows running the script from the examples directory
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from weaver import TableQA, WeaverConfig


def example_single_question_with_json_object():
    """Example: Ask a question using the new JSON object format."""
    print("\n=== Single Question Example (JSON Object) ===")
    
    # Configure Weaver
    config = WeaverConfig.from_env()
    qa = TableQA(config)
    
    # Create a question object (like from your datasets)
    question_obj =  {
        "table_id": "nu-0",
        "question": "which country had the most cyclists finish within the top 10?",
        "table_file_name": "./datasets/WikiTableQuestions/csv/203-csv/733.csv",
        "target_value": "Italy",
        "table_name": "2008 Clásica de San Sebastián"
    }

    # Ask the question
    result = qa.ask(question_obj)
    
    print(f"Question: {result.question}")
    print(f"Answer: {result.answer}")
    print(f"Correct: {result.is_correct}")


def example_batch_processing():
    """Example: Process multiple questions from a JSON file."""
    print("\n=== Batch Processing Example ===")
    
    config = WeaverConfig.from_env()
    qa = TableQA(config)
    
    # Show database backend being used
    print(f"Database Backend: {config.database.db_type.upper()}")
    print(f"Database: {config.database.db_name}")
    
    # Use the actual datasets directory (relative to parent directory)
    datasets_dir = Path(__file__).parent / "datasets"
    wikitq_path = datasets_dir / "wikitq.json"

    # Process a dataset (like wikitq.json format)
    results = qa.evaluate_dataset(
        dataset_name="wikitq",
        data_path=str(wikitq_path),
        num_samples=5  # Process first 25 questions
    )
    
    # Calculate metrics
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    accuracy = correct / total if total > 0 else 0
    
    print(f"Processed {total} questions")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    
    # Show individual results
    for i, result in enumerate(results):
        print(f"Q{i+1}: {result.question[:50]}...")
        print(f"A{i+1}: {result.answer}")
        print(f"Correct: {result.is_correct}")
        print()


def example_with_optional_context():
    """Example: Using optional context files."""
    print("\n=== Example with Optional Context ===")
    
    config = WeaverConfig.from_env()
    qa = TableQA(config)
    
    question_obj = {
        "table_id": "ADI/2011/page_61.pdf",
        "question": "what is the percentage change in cash flow hedges in 2011 compare to the 2010?",
        "table_file_name": "./datasets/FINQA/csv/ADI_2011_page_61.csv",
        "target_value": "9.9%",
        "table_name": "ADI/2011/page_61.pdf",
        "paragraphs": "undesignated hedges was $ 41.2 million and $ 42.1 million , respectively . the fair value of these hedging instruments in the company 2019s consolidated balance sheets as of october 29 , 2011 and october 30 , 2010 was immaterial . interest rate exposure management 2014 on june 30 , 2009 , the company entered into interest rate swap transactions related to its outstanding 5.0% ( 5.0 % ) senior unsecured notes where the company swapped the notional amount of its $ 375 million of fixed rate debt at 5.0% ( 5.0 % ) into floating interest rate debt through july 1 , 2014 . under the terms of the swaps , the company will ( i ) receive on the $ 375 million notional amount a 5.0% ( 5.0 % ) annual interest payment that is paid in two installments on the 1st of every january and july , commencing january 1 , 2010 through and ending on the maturity date ; and ( ii ) pay on the $ 375 million notional amount an annual three month libor plus 2.05% ( 2.05 % ) ( 2.42% ( 2.42 % ) as of october 29 , 2011 ) interest payment , payable in four installments on the 1st of every january , april , july and october , commencing on october 1 , 2009 and ending on the maturity date . the libor- based rate is set quarterly three months prior to the date of the interest payment . the company designated these swaps as fair value hedges . the fair value of the swaps at inception was zero and subsequent changes in the fair value of the interest rate swaps were reflected in the carrying value of the interest rate swaps on the balance sheet . the carrying value of the debt on the balance sheet was adjusted by an equal and offsetting amount . the gain or loss on the hedged item ( that is , the fixed-rate borrowings ) attributable to the hedged benchmark interest rate risk and the offsetting gain or loss on the related interest rate swaps for fiscal year 2011 and fiscal year 2010 were as follows : statement of income . the amounts earned and owed under the swap agreements are accrued each period and are reported in interest expense . there was no ineffectiveness recognized in any of the periods presented . the market risk associated with the company 2019s derivative instruments results from currency exchange rate or interest rate movements that are expected to offset the market risk of the underlying transactions , assets and liabilities being hedged . the counterparties to the agreements relating to the company 2019s derivative instruments consist of a number of major international financial institutions with high credit ratings . based on the credit ratings of our counterparties as of october 29 , 2011 , we do not believe that there is significant risk of nonperformance by them . furthermore , none of the company 2019s derivative transactions are subject to collateral or other security arrangements and none contain provisions that are dependent on the company 2019s credit ratings from any credit rating agency . while the contract or notional amounts of derivative financial instruments provide one measure of the volume of these transactions , they do not represent the amount of the company 2019s exposure to credit risk . the amounts potentially subject to credit risk ( arising from the possible inability of counterparties to meet the terms of their contracts ) are generally limited to the amounts , if any , by which the counterparties 2019 obligations under the contracts exceed the obligations of the company to the counterparties . as a result of the above considerations , the company does not consider the risk of counterparty default to be significant . the company records the fair value of its derivative financial instruments in the consolidated financial statements in other current assets , other assets or accrued liabilities , depending on their net position , regardless of the purpose or intent for holding the derivative contract . changes in the fair value of the derivative financial instruments are either recognized periodically in earnings or in shareholders 2019 equity as a component of oci . changes in the fair value of cash flow hedges are recorded in oci and reclassified into earnings when the underlying contract matures . changes in the fair values of derivatives not qualifying for hedge accounting are reported in earnings as they occur . the total notional amounts of derivative instruments designated as hedging instruments as of october 29 , 2011 and october 30 , 2010 were $ 375 million of interest rate swap agreements accounted for as fair value hedges and $ 153.7 million and $ 139.9 million , respectively , of cash flow hedges denominated in euros , british pounds and analog devices , inc . notes to consolidated financial statements 2014 ( continued ) ."
    }
    
    result = qa.ask(question_obj)
    
    print(f"Question: {result.question}")
    print(f"Answer: {result.answer}")


if __name__ == "__main__":
    # Make sure you have set your API key:
    # export OPENAI_API_KEY="your-key"
    # or
    # export GEMINI_API_KEY="your-key" 
    
    try:
        # Run examples
        
        
        # Uncomment to run actual examples (requires sample data)
        #example_single_question_with_json_object() 
        # example_batch_processing()
        example_with_optional_context()
        
        print("\n=== Setup Instructions ===")
        print("1. Set your LLM API key environment variable:")
        print("   export OPENAI_API_KEY='your-openai-key'")
        print()
        print("2. Prepare your data:")
        print("   - CSV/JSON table files")
        print("   - JSON dataset files (for batch processing)")
        print("   - Optional: column description files (.txt/.md)")
        print("   - Optional: table schema files (.json)")
        print()
        print("3. Run:")
        print("   python basic_usage.py")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. Set up your API key environment variable")
        print("2. Installed the weaver package")
        print("3. Created sample data files")
