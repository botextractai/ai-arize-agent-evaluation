# AI agent evaluation with Arize Phoenix

Artificial Intelligence (AI) agents often include many different components. It is therefore often challenging to evaluate and optimise the performance of AI agents by tuning the components, such as the prompt, the router (in this example a Large Language Model), the skills (tools), and the workflow.

This example shows how to run two structured experiments to evaluate an AI agent with Arize Phoenix. The results of this evaluation can then be used to further improve the AI agent.

Arize Phoenix is an open-source observability library designed for experimentation, evaluation, and troubleshooting. Arize Phoenix allows AI Engineers and Data Scientists to quickly visualise data, evaluate performance, track down issues, and export data to improve AI apps.

![alt text](https://github.com/user-attachments/assets/e5e948b9-7e2d-41e3-8d46-ba516d40616e "Arize Phoenix")

## Definition of an AI agent

AI agents are software based systems that can take actions on behalf of users by using reasoning. In this example, the reasoning is powered by an OpenAI Large Language Model (LLM). The LLM does the routing (interpreting the request and determining the correct tool). An AI agent does have one or multiple tools at its disposal, for example tools for executing program code, do internet searches, execute database queries, call Application Programming Interfaces (API's), or make LLM calls.

Agentic workflows usually involve multiple iterations (loops) of calling the router (as the "reason agent") and the tools.

## The example AI agent

The example AI agent is a data analysis assistant that can help you understand sales data from each of your stores. The sales data comes from a Parquet file in the `data` folder.

Steps in red are LLM calls, while steps in green are program executions.

![alt text](https://github.com/user-attachments/assets/34dfae78-79d0-4caa-a1a9-af5025b3f9b2 "Example Agent")

Skills (Tools):

- "Look Up Sales Data Tool": A data lookup skill to query your DuckDB database
- "Data Analysis Tool": A data analysis skill to draw conclusions from your data
- "Data Visualisation Tool": A data visualisation skill to generate graphs and visualisations about your data

The program code of the AI agent is in the `utils.my` file.

### Database

This example uses DuckDB as database. DuckDB supports direct querying of the Parquet data format. However, this example first reads the data in a Panda dataframe before converting the dataframe into an in-memory DuckDB table for SQL querying.

## Notes

You need an OpenAI API key for this example. [Get your OpenAI API key here](https://platform.openai.com/login). Insert the OpenAI API key into the `.env.example` file and then rename this file to just `.env` (remove the ".example" ending).

In addition to the OpenAI key, you also need a free Arize Phoenix API key for this example. [Get your free Arize Phoenix API key here](https://app.phoenix.arize.com/login/sign-up). You need to insert this same API key twice in the `.env` file, the first time for `OTEL_EXPORTER_OTLP_HEADERS`, and the second time for `PHOENIX_CLIENT_HEADERS`.

This example pauses the execution while displaying charts. Please close each chart window for the program execution to continue.
