# Warning control
import warnings
import datetime
import os

warnings.filterwarnings("ignore")

# Langchain
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_sambanova import ChatSambaNova


from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


from benchmark.evaluator import QAEvaluator
import json
import requests
import pandas as pd
from dotenv import load_dotenv
from app_config import config
import logging
logger = logging.getLogger(__name__)
from pathlib import Path

HERE = Path(__file__).parent.resolve()

path = "/".join(os.path.realpath(__file__).split("/")[0:-1])
env_path = path + "/.env"
if os.path.exists(env_path):
    print(f"'{env_path}' exists (could be a file or directory).")
    load_dotenv(env_path, override=True)
else:
    print(f"'{env_path}' does not exist. Look for environment variables")

TAPIS_CHATAGENT_BASE_URL = os.getenv("TAPIS_CHATAGENT_BASE_URL")
TAPIS_BASE_URL = os.getenv("TAPIS_BASE_URL")
TAPIS_USERNAME = os.getenv("TAPIS_USERNAME")
TAPIS_PASSWORD = os.getenv("TAPIS_PASSWORD")
headers={"Content-Type":"application/json"}
jsonbody={"username": TAPIS_USERNAME, "password": TAPIS_PASSWORD,"grant_type":"password"}
response = requests.post(TAPIS_BASE_URL+"/v3/oauth2/tokens", jsonbody, headers)
response_json = response.json()
#print(response_json)
tapis_token = response_json['result']['access_token']['access_token']
#print(tapis_token)


# For testing purpose
# Given 0llama pod is very slow, QNUM_MAX gives control to test only few iteration
# It is used to use only first QNUM_MAX number of questions from benchmark question answer set for the RAG pipeline
# QNUM_MAX = 0  allows the whole benchmark question answer set to be used for the pipeline

QNUM_MAX = config["max_questions_to_ask"]


def read_tapis_mcp_benchmark_file():
    qa_sets_path = config["benchmark_qa_input_to_mcp"]
    print(f"Reading input from file: {qa_sets_path}")
    with open(qa_sets_path, "r", encoding="utf-8") as f:
        qa_sets = json.load(f)
        return qa_sets


def invoke_chat_with_mcp(qa_sets, now):
    url = TAPIS_CHATAGENT_BASE_URL

    questions_answers = []
    i = 0
    for qa in qa_sets:
        i = i + 1
        if i == QNUM_MAX:
            print("Breaking loop at i =", i)
            break
        
        question = qa.get("question", "")
        print("\n question: " + question)
        
        jsonobj = {"message":question}
        tapis_headers = {'Content-type': 'application/json','x-tapis-token':tapis_token}
        response = requests.post(url, json = jsonobj, headers=tapis_headers)
        logger.info(response.text)
        logger.info(response.status_code)
        if response.status_code == 500:
            response = questions_answers.append({"status":"Internal Error"})
        elif response.status_code==502:
            response = questions_answers.append({"status":"Bad Gateway"})
        elif response.status_code==400:
            response = questions_answers.append({"status": "Bad Request"})
        else:
            response = json.loads(response.text)
        #answer = response["answer"]
        qa["answer"] = response["answer"]
        print("\n answer: " + response['answer'])
        questions_answers.append(qa)
    # write the outputs to a file
    # get the output path for MCP LLM generated answers which will be provided as input to benchmark evaluation
    qa_sets_mcp_llm_output_path = os.path.join(
        config["mcp_llm_generated_output_to_input_benchmark_eval"],
        f"mcp_generated_output_{now}.json",
    )
    print(f"Writing output to file: {qa_sets_mcp_llm_output_path}")
    with open(qa_sets_mcp_llm_output_path, "w") as f:
        json.dump(questions_answers, f, indent=4)

def get_llm_for_provider(provider):
    """
    Returns the LLM for this graph.
    """

    if provider == "openai":
        llm = ChatOpenAI(
            model=config["llm_name"],
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
    elif provider == "samba_nova":
        api_key = config["samba_nova_api_key"]
        print(f"Using SambaNova with api key {api_key[:5]}...{api_key[-5:]}")
        llm = ChatSambaNova(
            model=config["llm_name"],
            #sambanova_api_url=config["llm_base_url"],
            temperature=0,
            sambanova_api_key=api_key,
        )
    else:
        api_key = config["ollama_api_key"]
        llm = ChatOllama(
            model=config["llm_name"],
            base_url=config["llm_base_url"],
            temperature=0,
            api_key=api_key,
        )
    return llm

def chat_with_llm(llm):
    #general_system_template = """
    #You are a helpful support  who answers questions on Tapis v3.
    #"""
    general_system_template = """
    You are a research assistant answering questions on Tapis v3 based on Tapis Documentation and user information.
    Use the following context to answer the question at the end.
    Make sure not to make any changes to the context if possible when prepare answers  so as to provide accurate responses.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ----
    
    ----
    At the end of each answer it should cite metadata for relevant document as source.

    """
    general_user_template = "Question:```{message}```"
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(general_user_template),
    ]
    #qa_prompt = ChatPromptTemplate.from_messages(messages)
    chat_template = ChatPromptTemplate.from_messages(messages)
    chain = chat_template | llm

    return chain


def invoke_chat_with_base_llm(qa_sets, now):
    #url = config["llm_base_url"]
    #api_key = config["samba_nova_api_key"]
    llm_provider=config["llm_provider"]
    llm = get_llm_for_provider(llm_provider)
    questions_answers = []
    i = 0
    for qa in qa_sets:
        i = i + 1
        if i == QNUM_MAX:
            print("Breaking loop at i =", i)
            break
        
        question = qa.get("question", "")
        print("\n question: " + question)
        
        jsonobj = {"message":question}#"context":"Tapis v3"}
        #tapis_headers = {'Content-type': 'application/json','x-tapis-token':tapis_token}
        #response = requests.post(url, json = jsonobj, headers=tapis_headers)
        response = chat_with_llm(llm).invoke(jsonobj)
        logger.info(response)
        #logger.info(response.content)
        #logger.info(response.status_code)
        # if response.status_code == 500:
        #     response = questions_answers.append({"status":"Internal Error"})
        # elif response.status_code==502:
        #     response = questions_answers.append({"status":"Bad Gateway"})
        # elif response.status_code==400:
        #     response = questions_answers.append({"status": "Bad Request"})
        # else:
        #response = json.loads(response.content)
        #answer = response["answer"]
        #qa["answer"] = response["answer"]
        qa["answer"] = response.content
        #print("\n answer: " + response['answer'])
        questions_answers.append(qa)
    # write the outputs to a file
    # get the output path for MCP LLM generated answers which will be provided as input to benchmark evaluation
    qa_sets_mcp_llm_output_path = os.path.join(
        config["mcp_llm_generated_output_to_input_benchmark_eval"],
        f"mcp_generated_output_{now}.json",
    )
    print(f"Writing output to file: {qa_sets_mcp_llm_output_path}")
    with open(qa_sets_mcp_llm_output_path, "w") as f:
        json.dump(questions_answers, f, indent=4)

## evaluate the answers
def evaluate(run_question_answer, now):
    print(f"Starting the evaluation...")
    use_ollama = config["benchmark"]["use_ollama"]
    use_samba_nova = config["benchmark"]["use_samba_nova"]
    use_llm_judge = config["benchmark"]["use_llm_judge"]
    if use_ollama:
        model_name = config["benchmark"]["model_name_ollama"]
    elif use_samba_nova:
        model_name = config["benchmark"]["model_name_samba_nova"]
    else:
        model_name = config["benchmark"]["model_name_openai"]
    # If this execution included a run of the question/answer task, then the
    # input was generated by a prior step and will have the time stamp:
    if run_question_answer:
        qa_sets_rag_llm_output_path = os.path.join(
            config["mcp_llm_generated_output_to_input_benchmark_eval"],
            f"mcp_generated_output_{now}.json",
        )
    # Otherwise, the config should be a path to a file produced by a prior run:\
    else:
        qa_sets_rag_llm_output_path = config["mcp_llm_generated_output_to_input_benchmark_eval"]
    with open(
        qa_sets_rag_llm_output_path,
        "r",
        encoding="utf-8",
    ) as f:
        qa_pairs = json.load(f)
    
    print("model_name: " + model_name)
  
    # Instantiate and run evaluator
    evaluator = QAEvaluator(model_name=model_name, use_llm_judge=use_llm_judge)
    results = evaluator.evaluate(qa_pairs)
    # results = evaluator.evaluate(qa_pairs_subset)

    # Save results
    output_path = os.path.join(
        config["benchmark"]["eval_output"], f"qa_eval_output_{now}.csv"
    )
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nEvaluation completed. Results saved to '{output_path}'.")

    for r in results[:3]:  # print only 3 for brevity
        print("\nQ:", r["input"])
        print("Prediction:", r["output"])
        print("Reference:", r["reference"])
        print(
            "BLEU:",
            round(r["bleu"], 4),
            "| ROUGE-L:",
            round(r["rouge-l"], 4),
            "| CodeBERT:",
            #r["codebert"],
        )

def main():
    now = datetime.datetime.now().strftime("%Y-%m-%d.%H:%M:%S")
    run_question_answer = config["run_question_answer"]
    if run_question_answer:
        print("Reading Tapis MCP benchmark file")
        qa_sets = read_tapis_mcp_benchmark_file()
        qa_code_sets = []
        if config["code_generation_question_answer"]:
            print("Running code generation question/answer only task...")
            for question in qa_sets:
                if question['type'] ==   "Code Generation":
                    qa_code_sets.append(question)
            qa_sets = qa_code_sets

        else:
            print("Running question/answer task...")
        invoke_chat_with_mcp(qa_sets, now) #<============== Uncomment if want to use RAG
        #invoke_chat_with_base_llm(qa_sets, now)
    else:
        print("Not running question/answer task; config run_question_answer was False.")
    if config["run_evaluation"]:
        print("Running evaluator...")
        evaluate(run_question_answer, now)
    else:
        print("Not running evaluator; config run_evaluation was False.")

def read_tapis_mcp_benchmark_file():
    qa_sets_path = config["benchmark_qa_input_to_mcp"]
    print(f"Reading input from file: {qa_sets_path}")
    with open(qa_sets_path, "r", encoding="utf-8") as f:
        qa_sets = json.load(f)
        return qa_sets


    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s\t%(message)s")
    main()
    # now = datetime.datetime.now().strftime("%Y-%m-%d.%H:%M:%S")
    # qa_sets=read_tapis_mcp_benchmark_file()
    
    # convert_file(qa_sets,now)
    #replace_row_csv()