from datetime import datetime
from phoenix.evals import OpenAIModel, llm_classify, TOOL_CALLING_PROMPT_TEMPLATE
from phoenix.experiments import run_experiment
from phoenix.experiments.types import Example
from utils import get_phoenix_endpoint, run_agent, tools
from utils import process_messages, update_sql_gen_prompt
import json
import nest_asyncio
import pandas as pd
import phoenix as px

import warnings
warnings.filterwarnings('ignore')

nest_asyncio.apply()


# Creating the dataset of test questions
# --------------------------------------

eval_model=OpenAIModel(model="gpt-4o")

px_client = px.Client()

overall_experiment_questions = [
    {'question': 'What was the most popular product SKU?',
     'sql_result': '   SKU_Coded  Total_Qty_Sold 0    6200700         52262.0',
     'sql_generated': '```sql\nSELECT SKU_Coded, SUM(Qty_Sold) AS Total_Qty_Sold\nFROM sales\nGROUP BY SKU_Coded\nORDER BY Total_Qty_Sold DESC\nLIMIT 1;\n```'},
    {'question': 'What was the total revenue across all stores?',
     'sql_result': '   Total_Revenue 0   1.327264e+07',
     'sql_generated': '```sql\nSELECT SUM(Total_Sale_Value) AS Total_Revenue\nFROM sales;\n```'},
    {'question': 'Which store had the highest sales volume?',
     'sql_result': '   Store_Number  Total_Sales_Volume 0          2970             59322.0',
     'sql_generated': '```sql\nSELECT Store_Number, SUM(Total_Sale_Value) AS Total_Sales_Volume\nFROM sales\nGROUP BY Store_Number\nORDER BY Total_Sales_Volume DESC\nLIMIT 1;\n```'},
    {'question': 'Create a bar chart showing total sales by store',
     'sql_result': '    Store_Number    Total_Sales 0            880  420302.088397 1           1650  580443.007953 2           4180  272208.118542 3            550  229727.498752 4           1100  497509.528013 5           3300  619660.167018 6           3190  335035.018792 7           2970  836341.327191 8           3740  359729.808228 9           2530  324046.518720 10          4400   95745.620250 11          1210  508393.767785 12           330  370503.687331 13          2750  453664.808068 14          1980  242290.828499 15          1760  350747.617798 16          3410  410567.848126 17           990  378433.018639 18          4730  239711.708869 19          4070  322307.968330 20          3080  495458.238811 21          2090  309996.247965 22          1320  592832.067579 23          2640  308990.318559 24          1540  427777.427815 25          4840  389056.668316 26          2860  132320.519487 27          2420  406715.767402 28           770  292968.918642 29          3520  145701.079372 30           660  343594.978075 31          3630  405034.547846 32          2310  412579.388504 33          2200  361173.288199 34          1870  401070.997685',
     'sql_generated': '```sql\nSELECT Store_Number, SUM(Total_Sale_Value) AS Total_Sales\nFROM sales\nGROUP BY Store_Number;\n```'},
    {'question': 'What percentage of items were sold on promotion?',
     'sql_result': '   Promotion_Percentage 0              0.625596',
     'sql_generated': "```sql\nSELECT \n    (SUM(CASE WHEN On_Promo = 'Yes' THEN 1 ELSE 0 END) * 100.0) / COUNT(*) AS Promotion_Percentage\nFROM \n    sales;\n```"},
    {'question': 'What was the average transaction value?',
     'sql_result': '   Average_Transaction_Value 0                  19.018132',
     'sql_generated': '```sql\nSELECT AVG(Total_Sale_Value) AS Average_Transaction_Value\nFROM sales;\n```'},
    {'question': 'Create a line chart showing sales in 2021',
     'sql_result': '  sale_month  total_quantity_sold  total_sales_value 0 2021-11-01              43056.0      499984.428193 1 2021-12-01              75724.0      910982.118423',
     'sql_generated': '```sql\nSELECT MONTH(Sold_Date) AS Month, SUM(Total_Sale_Value) AS Total_Sales\nFROM sales\nWHERE YEAR(Sold_Date) = 2021\nGROUP BY MONTH(Sold_Date)\nORDER BY MONTH(Sold_Date);\n```'}
]

overall_experiment_df = pd.DataFrame(overall_experiment_questions)

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# create a dataset consisting of input questions and expected outputs
dataset = px_client.upload_dataset(dataframe=overall_experiment_df,
                                   dataset_name=f"overall_experiment_inputs-{now}",
                                   input_keys=["question"],
                                   output_keys=["sql_result", "sql_generated"])


# Link to Arize Phoenix UI
# ------------------------

print(get_phoenix_endpoint())


# Setting up the evaluators
# -------------------------
# Template prompts for the LLM-as-a-judge that are used to evaluate the clarity of the 
# analysis of tool 2 and the correctness of the entities mentioned in the analysis

CLARITY_LLM_JUDGE_PROMPT = """
In this task, you will be presented with a query and an answer. Your objective is to evaluate the clarity 
of the answer in addressing the query. A clear response is one that is precise, coherent, and directly 
addresses the query without introducing unnecessary complexity or ambiguity. An unclear response is one 
that is vague, disorganized, or difficult to understand, even if it may be factually correct.

Your response should be a single word: either "clear" or "unclear," and it should not include any other 
text or characters. "clear" indicates that the answer is well-structured, easy to understand, and 
appropriately addresses the query. "unclear" indicates that the answer is ambiguous, poorly organized, or 
not effectively communicated. Please carefully consider the query and answer before determining your 
response.

After analyzing the query and the answer, you must write a detailed explanation of your reasoning to 
justify why you chose either "clear" or "unclear." Avoid stating the final label at the beginning of your 
explanation. Your reasoning should include specific points about how the answer does or does not meet the 
criteria for clarity.

[BEGIN DATA]
Query: {query}
Answer: {response}
[END DATA]
Please analyze the data carefully and provide an explanation followed by your response.

EXPLANATION: Provide your reasoning step by step, evaluating the clarity of the answer based on the query.
LABEL: "clear" or "unclear"
"""

ENTITY_CORRECTNESS_LLM_JUDGE_PROMPT = """
In this task, you will be presented with a query and an answer. Your objective is to determine whether all 
the entities mentioned in the answer are correctly identified and accurately match those in the query. An 
entity refers to any specific person, place, organization, date, or other proper noun. Your evaluation 
should focus on whether the entities in the answer are correctly named and appropriately associated with 
the context in the query.

Your response should be a single word: either "correct" or "incorrect," and it should not include any 
other text or characters. "correct" indicates that all entities mentioned in the answer match those in the 
query and are properly identified. "incorrect" indicates that the answer contains errors or mismatches in 
the entities referenced compared to the query.

After analyzing the query and the answer, you must write a detailed explanation of your reasoning to 
justify why you chose either "correct" or "incorrect." Avoid stating the final label at the beginning of 
your explanation. Your reasoning should include specific points about how the entities in the answer do or 
do not match the entities in the query.

[BEGIN DATA]
Query: {query}
Answer: {response}
[END DATA]
Please analyze the data carefully and provide an explanation followed by your response.

EXPLANATION: Provide your reasoning step by step, evaluating whether the entities in the answer are 
correct and consistent with the query.
LABEL: "correct" or "incorrect"
"""

# evaluator for the router
def function_calling_eval(input: str, output: str) -> float:
    if output is None:
        return 0
    function_calls = output.get("tool_calls")
    if function_calls:
        eval_df = pd.DataFrame({
            "question": [input.get("question")] * len(function_calls),
            "tool_call": function_calls
        })

        tool_call_eval = llm_classify(
            data = eval_df,
            template = TOOL_CALLING_PROMPT_TEMPLATE.template[0].template.replace("{tool_definitions}",
                                                                                 json.dumps(tools).replace("{", '"').replace("}", '"')),
            rails = ['correct', 'incorrect'],
            model=eval_model,
            provide_explanation=True
        )

        tool_call_eval['score'] = tool_call_eval.apply(lambda x: 1 if x['label']=='correct' else 0, axis=1)
        return tool_call_eval['score'].mean()
    else:
        return 0

# evaluator for tool 1: database lookup
def evaluate_sql_result(output, expected) -> bool:
    if output is None:
        return False
    sql_result = output.get("tool_responses")
    if not sql_result:
        return True
    
    # find first lookup_sales_data response
    sql_result = next((r for r in sql_result if r.get("tool_name") == "lookup_sales_data"), None)
    if not sql_result:
        return True
    
    # get the first response
    sql_result = sql_result.get("tool_response", "")
    
    # extract just the numbers from both strings
    result_nums = ''.join(filter(str.isdigit, sql_result))
    expected_nums = ''.join(filter(str.isdigit, expected.get("sql_result")))
    return result_nums == expected_nums

# evaluator for tool 2: data analysis (clarity)
def evaluate_clarity(output: str, input: str) -> bool:
    if output is None:
        return False
    df = pd.DataFrame({"query": [input.get("question")],
                       "response": [output.get("final_output")]})
    response = llm_classify(
        data=df,
        template=CLARITY_LLM_JUDGE_PROMPT,
        rails=["clear", "unclear"],
        model=eval_model,
        provide_explanation=True
    )
    return response['label'] == 'clear'

# evaluator for tool 2: data analysis (entity correctness)
def evaluate_entity_correctness(output: str, input: str) -> bool:
    if output is None:
        return False
    df = pd.DataFrame({"query": [input.get("question")],
                       "response": [output.get("final_output")]})
    response = llm_classify(
        data=df,
        template=ENTITY_CORRECTNESS_LLM_JUDGE_PROMPT,
        rails=["correct", "incorrect"],
        model=eval_model,
        provide_explanation=True
    )
    return response['label'] == 'correct'

# evaluator for tool 3: data visualization
def code_is_runnable(output: str) -> bool:
    """Check if the code is runnable"""
    if output is None:
        return False
    generated_code = output.get("tool_responses")
    if not generated_code:
        return True
    
    # find first lookup_sales_data response
    generated_code = next((r for r in generated_code if r.get("tool_name") == "generate_visualization"), None)
    if not generated_code:
        return True
    
    # get the first response
    generated_code = generated_code.get("tool_response", "")
    generated_code = generated_code.strip()
    generated_code = generated_code.replace("```python", "").replace("```", "")
    try:
        exec(generated_code)
        return True
    except Exception as e:
        return False


# Defining the task
# -----------------

def run_agent_task(example: Example) -> str:
    print("Starting agent with messages:", example.input.get("question"))
    messages = [{"role": "user", "content": example.input.get("question")}]
    ret = run_agent(messages)
    return process_messages(ret)


# Running the experiment
# ----------------------

experiment = run_experiment(dataset,
                            run_agent_task,
                            evaluators=[function_calling_eval,
                                        evaluate_sql_result,
                                        evaluate_clarity,
                                        evaluate_entity_correctness,
                                        code_is_runnable],
                            experiment_name="Overall Experiment - Version 1",
                            experiment_description="Evaluating the overall experiment")


# Running the experiment - change in prompt
# -----------------------------------------

new_prompt = """
Generate an SQL query based on a prompt.
Do not reply with anything besides the SQL query.
The prompt is: {prompt}

The available columns are: {columns}
The table name is: {table_name}

Think before you respond.
"""

update_sql_gen_prompt(new_prompt)

experiment = run_experiment(dataset,
                            run_agent_task,
                            evaluators=[function_calling_eval,
                                        evaluate_sql_result,
                                        evaluate_clarity,
                                        evaluate_entity_correctness,
                                        code_is_runnable],
                            experiment_name="Overall Experiment - Version 2",
                            experiment_description="Evaluating the overall experiment, with changes to sql prompt")
