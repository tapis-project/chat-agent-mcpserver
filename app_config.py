import os
from dotenv import load_dotenv
from pathlib import Path

HERE = Path(__file__).parent.resolve()
PARENT = Path(__file__).parent.parent.resolve()

# load .env file to environment
# get the path to .env file
path = "/".join(os.path.realpath(__file__).split("/")[0:-1])
env_path = path + "/.env"
if os.path.exists(env_path):
    print(f"'{env_path}' exists (could be a file or directory).")
    load_dotenv(env_path, override=True)
else:
    print(f"'{env_path}' does not exist. Look for environment variables")


# get all the environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
SAMBANOVA_BASE_URL = os.getenv("SAMBANOVA_BASE_URL")
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "https://ollama.pods.tacc.develop.tapis.io"
)

TAPIS_MCP_SERVER_BASE_URL = os.getenv("TAPIS_MCP_SERVER_BASE_URL")
PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY")
PHOENIX_COLLECTOR_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")

"""
config = {
    # For Tejas/SambaNova ---
    "llm_provider": "samba_nova",
    #"llm_name": "Meta-Llama-3.1-405B-Instruct",
    "llm_name":"Llama-4-Maverick-17B-128E-Instruct",
    "llm_base_url": SAMBANOVA_BASE_URL,
    "samba_nova_api_key": SAMBANOVA_API_KEY,
    "benchmark_qa_input_to_rag": os.path.join(HERE, "data/LLM_generated_v2.json"),
    # There are two options here; if running the question/answer task as part of this run,
    # this should be a *directory* that already exists and the program will generate a file
    # name that includes a time stamp for the run, e.g.,
    "rag_llm_generated_output_to_input_benchmark_eval": os.path.join(HERE, "data"),
    # Alternatively, if just running the evaluator, specify a path to a previously generated file, e.g.,
    # "rag_llm_generated_output_to_input_benchmark_eval": os.path.join(HERE, "data", "LLM_generated_v2.json"),
    # Evaluation of the Results -------
    # Whether to run the evaluation of the results (QAEvaluator)
    "run_evaluation": True,
    "benchmark": {
        "use_ollama": False,
        "use_samba_nova": True,
        "use_llm_judge": True,
        # For Ollama Pods
        "model_name_ollama": "llama3.1:8b",
        "model_name_openai": "gpt-5.1",
        #"model_name_samba_nova": "Meta-Llama-3.1-405B-Instruct",
        "model_name_samba_nova": "Llama-4-Maverick-17B-128E-Instruct",
        "qa_sets_path": os.path.join(HERE, "data/mcp_llm_generated_output.json"),
        # Should be a directory that already exists; file will include timestamp.
        "eval_output": os.path.join(
            HERE,
            "data/outputs",
        ),
    },
}
"""

"""
use this config for openai
"""
config= {
    "tapis_mcp_base_url": TAPIS_MCP_SERVER_BASE_URL,
    "llm_provider": "openai",#"samba_nova",
    "llm_name": "gpt-5.1", #"gpt-4o",#"Llama-4-Maverick-17B-128E-Instruct",#"gpt-5.1", #"gpt-4o",
    #"llm_base_url": #"https://api.openai.com/v1",
    "openai_api_key":OPENAI_API_KEY,
    "run_question_answer": False,
    "max_questions_to_ask": 0,
    "code_generation_question_answer": False,
   # "benchmark_qa_input_to_mcp": os.path.join(HERE, "data/data_new_systems_apps_all.json"), ### <=== Change the file name according to the requirement
    "benchmark_qa_input_to_mcp": os.path.join(HERE, "data/mcp_qa.json"), ### <=== Change the file name according to the requirement
#     # There are two options here; if running the question/answer task as part of this run,
#     # this should be a *directory* that already exists and the program will generate a file
#     # name that includes a time stamp for the run, e.g.,
    #"mcp_llm_generated_output_to_input_benchmark_eval": os.path.join(HERE, "data"),
#     # Alternatively, if just running the evaluator, specify a path to a previously generated file, e.g.,
"mcp_llm_generated_output_to_input_benchmark_eval": os.path.join(HERE, "data","rag_userspecific_openai_mcp_generated_output_2026-06-04.19:42:20.json"),
#    "mcp_llm_generated_output_to_input_benchmark_eval": os.path.join(HERE, "data", "user_specific_openai_generated_no_rag_output_2026-06-04.16:27:32.json"),
#"mcp_llm_generated_output_to_input_benchmark_eval": os.path.join(HERE, "data","llama4_generated_user_specific_noragoutput_2026-06-04.16:06:02.json"),
#"mcp_llm_generated_output_to_input_benchmark_eval": os.path.join(HERE, "data","llama4_gnereted_answers_no_rag_output_2026-06-04.13:05:35.json"),
   #"mcp_llm_generated_output_to_input_benchmark_eval": os.path.join(HERE, "data","openai_generated_answers_no_rag_output_2026-06-04.12:09:01.json"),
    #  "mcp_llm_generated_output_to_input_benchmark_eval": os.path.join(HERE, "data","user_specific_mcp_generated_output_2026-06-04.06:36:34.json"),
    # "mcp_llm_generated_output_to_input_benchmark_eval": os.path.join(HERE, "data", "new_sambanova_mcp_generated_output_2026-05-20.16:40:06.json"),
    #"mcp_llm_generated_output_to_input_benchmark_eval": os.path.join(HERE, "data", "new_openai_mcp_generated_output_2026-05-20.16:08:05.json"),
# # Evaluation of the Results -------
#     # Whether to run the evaluation of the results (QAEvaluator)
    "run_evaluation": True,
    "llm_base_url": SAMBANOVA_BASE_URL,
    "samba_nova_api_key": SAMBANOVA_API_KEY,
    "benchmark": {
        "use_ollama": False,
        "use_samba_nova": False,
        "use_llm_judge": True,
        # For Ollama Pods
        "model_name_ollama": "llama3.1:8b",
        "model_name_openai": "gpt-5.1",
        "model_name_samba_nova": "Llama-4-Maverick-17B-128E-Instruct",
        #"qa_sets_path": os.path.join(HERE, "data/user_specific_mcp_generated_output_2026-06-04.06:36:34.json"),
        #"qa_sets_path": os.path.join(HERE, "new_sambanova_mcp_generated_output_2026-05-20.16:40:06.json"),
        #"qa_sets_path": os.path.join(HERE, "new_openai_mcp_generated_output_2026-05-20.16:08:05.json"),
        # Should be a directory that already exists; file will include timestamp.
        "qa_sets_path": os.path.join(HERE, "data/mcp_generated_output.json"),
        "eval_output": os.path.join(
            HERE,
            "data/outputs",
        ),
    },
    "phoenix_api_key": PHOENIX_API_KEY,
    "phoenix_collector_endpoint":PHOENIX_COLLECTOR_ENDPOINT
}
#"""


