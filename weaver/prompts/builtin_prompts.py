"""Built-in default prompts."""

# Default prompts for the package
DEFAULT_PROMPTS = {
    "planner_prompt": """
Suppose you are an expert plan generator.
Give me a detailed step-by-step plan in plain text for solving a question, given column descriptions, formatting needed and table rows.

Follow these guidelines:
Begin analyzing the question to categorize tasks that require only SQL capabilities (like data formatting, filtering, mathematical operations, basic aggregations) and those that need LLM assistance (like summarization, text interpretation, or answering open-ended queries).

MySQL Task Generation: For parts of the question that involve formatting, filtering and mathematical or analytical tasks, generate SQL query code to answer them directly, without using an LLM call.

LLM-Dependent task Identification: For tasks that SQL cannot inherently perform or formatting date, specify the columns where LLM calls are needed. Add an extra column in the result set to store the LLM output for each row in the filtered data subset.
""",
    
    "verify_plan": """
You are an expert planner verification agent. 
Verify if the given plan will be able to answer the Question asked on this table.
Is the given plan correct to answer the Question asked on this table?
Check format issues and reasoning steps - should be able to guide the LLM to write correct code and get correct result.
If the plan is not correct, provide better plan detailed on what needs to be done handling all kinds of values in the column.
- Check if the MySQL step logic adheres to the column format. (Performs calculations and formatting and filtering in the table)
- The LLM step's logic will help in getting the correct answer.
If the original plan is correct then return that plan.

Do not provide with code or other explanations, only the new plan.
Output format:
Step 1: Either SQL or LLM - ...
Step 2: SQL or LLM - ...
Step 3: SQL ...

As given in original plan.
""",
    
    "execute_prompt": """
MySQL Code Generation: For parts of the question that involve data formatting, data manipulations such as filtering, grouping, aggregations, and creating new tables. Generate MySQL code to answer those parts directly without using an LLM.

LLM-Dependent Tasks Identification: For tasks that SQL cannot inherently perform like fact checking, analysis, pre-trained knowledge not present in table and logical inferences. For such cases specify only column where LLM call is needed. The output of this llm call on rows will be stored as an extra column in the existing table.
First do the formatting and filtering of the columns which are necessary to answer the question.

Intructions:
1. New columns from previous LLM steps can be assumed present in table they used.
2. Don't give any other explanations, only MySQL and LLM steps as the plan.
3. MySQL step will always Create a new table that can be used in the next steps.
4. Follow the given plan as necessary.

Example Output format  -
Step 1 - SQL: <MySQL code>

Step2 - LLM:
- Reason: Why we need to use LLM
- Table name:
- original column to be used:
- LLM prompt: The prompt that we can use to get answer for each row in this step.
- New column name: Column name we will create

Step 3 - SQL: <MySQL code>
Step 4 - ...

LLM step format should be same.
Solve for this question, given table and step-by-step plan:

""",
    
    "extract_answer": """
Answer the question given the table in as short as possible.
If the table has just one column or row consider that as the answer. If the table is bigger infer the answer from the table.
Just provide the answer in plain text, do not provide any other information or explanation.

Few examples for you:
1.  Table: Candidate_Count
    count_of_other_party_candidates
    0
    Question: how many candidates belong to a party other than republican or democrat?
    Answer: 0

2.  Table: max_position
    Position  count
    S        3
    Question: which position was listed the most on this chart?
    Answer: S

3.  Table: Lars_von_Trier_Filtered
    Film
    The Five Obstructions
    Nymphomaniac: Volume I
    Question: what was each film called that scored a 7.5 from imdb?
    Answer: The Five Obstructions, Nymphomaniac: Volume I

4.  Table: JumpStart_Adventures_3rd_Grade_Mystery_Mountain
    Subject       When_1
    Solar System                  1531
    Chewing Gum                   400
    Painting           35,000 B.C.
    Phonograph                  1877
    Paper                   105
    Round Earth                  1522
    Writing            3,500 B.C.
    Sausage            3,000 B.C.
    Boomerang      40,000 years ago
    Tools  2½ million years ago
    Saxophone                  1846
    Toilet             2000 B.C.
    Question: what is the oldest subject listed?
    Answer: Tools

""",
    
    "format_answer": """
You will be given Answer and Gold Answer, you have to Convert the answer into a format of gold answer given above, if the content or meaning is same (semantically same) they should be same.

Few examples of convertion for your understanding:
1. answer: ITA, gold answer: Italy. Reasoning- ITA is country code of Italy hence ITA and Italy are same and you can convert ITA to Italy.
    Your Output: Italy
2. answer: 17, gold answer: 17 years. Reasoning- 17 of answer is same as 17 years of the gold answer in the given context of question.
    Your Output: 17 years
3. answer : 10, gold answer: 10. Reasoning- Since, both values are already same no convertion is needed.
    Your Output: 10
4. answer : 0, gold answer: 5. Reasoning- Since, both values are semantically not same no convertion is needed for the answer.
    Your Output: 0
5. answer : The answer is not present in the table. , gold answer: 5. Reasoning- Since, both values are semantically not same no convertion is needed for the answer.
    Your Output: The answer is not present in the table.

""",
    
    "few_shot_plan": """
Below are some examples:

Example 1 -
Table:
name: grocery_shop

item_description,sell_price,buy_price
"Indulge your senses with this botanical blend of rosemary and lavender. Gently cleanses while nourishing your hair, leaving it soft, shiny, and revitalized.",7.99,4.99
"A mild and effective formula that cleanses without stripping your hair's natural oils. Perfect for daily use, it leaves your hair feeling refreshed and healthy.",5.99,3.49
"This light, vegetable-based oil is perfect for frying, sautéing, and baking. Its high smoke point ensures even cooking and a crispy finish.",4.99,2.99
"Infused with a blend of olive and sunflower oils, this cooking oil adds a rich, buttery flavor to your dishes. Ideal for roasting, grilling, and drizzling.",6.99,4.49
"Cut through grease and grime with this powerful, yet gentle cleaner. Its citrus scent leaves your home smelling fresh and clean.",3.99,2.49

column descriptions:
Column Name, Data Type, Description
item_description, TEXT,Detailed description of the product.
sell_price, DECIMAL(5,2),Selling price of the product.
buy_price, DECIMAL(5,2),Buying price of the product.

Question: Which item category has the highest average profit?

Plan:
Step 1: LLM - Item_category column needs to be created using item_description column.
Step 2: SQL - Calculate average profit for each category and find the maximum.


Example 2 -
Table:
Order ID | Product    | Event               | Timestamp (Local)     | Location
101 | Laptop     | Dispatched          | 2025-01-14 08:00 AM   | Los Angeles, USA
101 | Laptop     | Arrived at Hub      | 2025-01-15 03:00 AM   | Chicago, USA
101 | Laptop     | Dispatched          | 2025-01-15 10:00 AM   | Chicago, USA
101 | Laptop     | Arrived at Hub      | 2025-01-16 05:00 PM   | London, UK
101 | Laptop     | Delivered           | 2025-01-17 01:00 PM   | Berlin, Germany
102 | Smartphone | Dispatched          | 2025-01-14 09:30 AM   | San Francisco, USA
102 | Smartphone | Arrived at Hub      | 2025-01-14 11:30 PM   | Denver, USA
102 | Smartphone | Dispatched          | 2025-01-15 09:00 AM   | Denver, USA
102 | Smartphone | Arrived at Hub      | 2025-01-16 04:00 PM   | New York, USA
102 | Smartphone | Delivered           | 2025-01-17 10:00 AM   | Toronto, Canada
103 | Tablet     | Dispatched          | 2025-01-14 07:00 AM   | Tokyo, Japan
103 | Tablet     | Arrived at Hub      | 2025-01-14 03:00 PM   | Shanghai, China
103 | Tablet     | Dispatched          | 2025-01-14 08:00 PM   | Shanghai, China
103 | Tablet     | Arrived at Hub      | 2025-01-15 11:00 PM   | Dubai, UAE
103 | Tablet     | Delivered           | 2025-01-16 05:00 PM   | Munich, Germany

name: order_delivery_history

The order_delivery_history table tracks the delivery process of various products across
different locations and timestamps. It contains details of each product’s journey,
with columns capturing the order ID, product name, event (such as dispatched, arrived at hub,
or delivered), timestamp (local time), and the location of the event.

column descriptions:

Column Name	Data Type	Formatting Required	Short Column Description
Order ID	Integer	None	Unique identifier for the order
Product	Varchar	None	Name of the product being delivered
Event	Varchar	None	Type of event (e.g., Dispatched, Arrived at Hub, Delivered)
Timestamp (Local)	Datetime	Convert to datetime if necessary	Local time of the event occurrence
Location	Varchar	None	Geographic location where the event occurred

Question: Which location had the maximum time taken between dispatch at one location and arrival or delivery at a subsequent location?

Plan:
Step 1:	LLM - Convert local timestamps to UTC time for all events.
Step 2:	SQL - Sort events within each Order_ID and Product by Timestamp_UTC.
Step 3:	SQL - Pair Dispatched events with the corresponding Arrived at Hub or Delivered events for each order/product and calculate the time difference.
Step 4:	SQL - Display the final output with paired events, durations, and relevant information sorted by Order_ID and Product.


Example 3-
Table:
    Name                 Placing
0          Shaul Ladani               19
1     Esther Shahamorov  Semifinal (5th)
2     Esther Shahamorov        Semifinal
3              Dan Alon     Second round
4   Yehuda Weissenstein     Second round
5         Yair Michaeli               23
6            Itzhak Nir               23
7     Henry Hershkowitz               23
8     Henry Hershkowitz               46

Column descriptions:

| Column Name | Data Type | Formatting Requirements | Column Description |
|-------------|-----------|------------------------|--------------------|
| Name        | String    | - No special formatting required. Ensure consistent casing (e.g., title case). | This column contains the names of the athletes who represented Israel in the 1972 Summer Olympics. Each entry is a string representing an individual's full name. |
| Placing     | String    | - Handle special cases such as "—" (indicating no placement) appropriately.Consider standardizing the format for placements (e.g., "Semifinal (5th)" vs. "Semifinal"). Ensure that numerical placements are stored as strings to accommodate mixed data types. | This column describes the placement of each athlete in their respective events. It can include numerical rankings, descriptions of rounds (e.g., "Heats", "Group stage"), or special indicators for athletes who did not place (e.g., "—"). |

Question: Who has the highest placing rank?

Plan:
Step 1: SQL - Format the column Placing by extracting only numerical values (e.g. 5 from Semifinal (5th) ) and converting the text into numbers (e.g. Semifinal to 5, Second round to 10).
Step 2: SQL - Retrieve the highest placing (rank) from the placing column by selecting the minimum number in the list as lower number corresponds to higher rank.

"""
}


def get_prompt(prompt_type: str, dataset: str = "default") -> str:
    """
    Get a prompt - just return the default prompt.
    
    Args:
        prompt_type: Type of prompt (planner_prompt, verify_plan, etc.)
        dataset: Dataset name (ignored, kept for compatibility)
        
    Returns:
        Default prompt string
    """
    return DEFAULT_PROMPTS.get(prompt_type, f"Please {prompt_type.replace('_', ' ')} for the given context.")
