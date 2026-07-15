import os
import warnings

# # Disable all OpenTelemetry exporters
# os.environ["OTEL_TRACES_EXPORTER"] = "none"
# os.environ["OTEL_METRICS_EXPORTER"] = "none"
# os.environ["OTEL_LOGS_EXPORTER"] = "none"

# # Disable OTLP explicitly (belt + suspenders)
# os.environ["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] = "none"
# os.environ["OTEL_EXPORTER_OTLP_METRICS_PROTOCOL"] = "none"
# os.environ["OTEL_EXPORTER_OTLP_LOGS_PROTOCOL"] = "none"

# os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006/v1/traces"

# # Suppress the OTel "Overriding current TracerProvider" logs
# warnings.filterwarnings("ignore", message="Overriding of current TracerProvider")

# os.environ["OTEL_PYTHON_DISABLED_INSTRUMENTATIONS"] = "all"
# # 3. Disable the standard OTEL ports to stop "Connection Refused"
# os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = "http://localhost:6006/v1/traces"
# # 4. Remove any existing endpoint for metrics just to be safe
# if "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT" in os.environ:
#     del os.environ["OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"]
import sys
import asyncio
import json
import re
import pandas as pd
import nltk
from pathlib import Path
from dotenv import load_dotenv

# NLP Metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import code_bert_score

# AI & Tracing
from openai import OpenAI
from phoenix.otel import register
from phoenix.evals import evaluate_dataframe, LLM
from phoenix.evals.metrics import CorrectnessEvaluator, FaithfulnessEvaluator
from opentelemetry.propagate import set_global_textmap

# App config and Agent
from app_config import *


load_dotenv()
nltk.download("punkt", quiet=True)

class QAEvaluator:
    def __init__(self, model_name="gpt-4o-mini", use_llm_judge=True):
        self.model_name = model_name
        self.use_llm_judge = use_llm_judge
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        
        # Unified Phoenix LLM
        #self.llm = LLM(provider="openai", model=model_name)
        self.llm = LLM(provider="openai", model="gpt-5.1")
        #self.llm = LLM(provider="litellm", model=model_name,base_url=config["llm_base_url"],api_key=config["samba_nova_api_key"],)
        
        # In 3.0, parameters like provide_explanation live in the Evaluator object
        self.correctness_eval = CorrectnessEvaluator(
            llm=self.llm)
        
        # Create Client for manual/legacy judge
        if config["benchmark"].get("use_ollama"):
            self.client = OpenAI(base_url=config["llm_base_url"], api_key="ollama")
        elif config["benchmark"].get("use_samba_nova"):
            self.client = OpenAI(base_url=config["llm_base_url"], api_key=config["samba_nova_api_key"])
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def ask_llm(self, question, prediction, reference):
        prompt = (
            f"Question: {question}\n"
            f"Predicted Answer: {prediction}\n"
            f"Reference Answer: {reference}\n\n"
            "Evaluate how well the predicted answer answers the question compared to the reference.\n"
            "Give a score between 1 (poor) and 5 (excellent) and explain briefly.\n"
            "Respond only with: Score: X\nExplanation: ..."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            reply = response.choices[0].message.content
            score_match = re.search(r"Score:\s*(\d+)", reply)
            score = int(score_match.group(1)) if score_match else None
            return score, reply
        except Exception as e:
            print(f"LLM Error: {e}")
            return None, "Error calling LLM"

    def evaluate(self, qa_pairs):
        results = []
        codebert_predictions, codebert_references, codebert_indices = [], [], []

        for idx, qa in enumerate(qa_pairs):
            prediction = qa.get("answer", "")
            reference = qa.get("reference", "")
            question = qa.get("question", "")
            qtype = qa.get("type", "").lower()
            service = qa.get("service","")

            result = {
                "input": question,
                "output": prediction,
                "reference": reference,
                "question_type": qtype,
                "service":service,
            }

            # NLP Metrics
            if qtype in ["conceptual", "yes/no","user-specific"] and prediction and reference:
                bleu = sentence_bleu([reference.split()], prediction.split(), smoothing_function=self.smooth)
                rouge_scores = self.rouge.get_scores(prediction, reference)[0]
                result.update({
                    "bleu": bleu,
                    "rouge-1": rouge_scores["rouge-1"]["f"],
                    "rouge-l": rouge_scores["rouge-l"]["f"],
                })
            elif qtype in ["code generation", "code correction"]:
                codebert_predictions.append(prediction)
                codebert_references.append(reference)
                codebert_indices.append(idx)
            
            results.append(result)

        if codebert_predictions:
            _, _, F1, *_ = code_bert_score.score(cands=codebert_predictions, refs=codebert_references, lang="python")
            for i, res_idx in enumerate(codebert_indices):
                results[res_idx]["codebert_f1"] = float(F1[i])

        df = pd.DataFrame(results)
        print("🚀 Running Phoenix Correctness Evaluator...")
        
        eval_results_df = evaluate_dataframe(
            dataframe=df,
            evaluators=[self.correctness_eval]
        )
        
        final_df = pd.concat([df, eval_results_df], axis=1)
        final_results_list = final_df.to_dict(orient="records")
        
        if self.use_llm_judge:
            print("⚖️ Running Manual LLM Judge...")
            for r in final_results_list:
                m_score, m_expl = self.ask_llm(r["input"], r["output"], r["reference"])
                r["llm_score"] = m_score
                r["lm_explanation"] = m_expl

        return final_results_list

# def load_data(json_path):
#     with open(json_path, "r") as f:
#         return json.load(f)

# async def run_benchmark():
#     # Register only the project. The ENV vars handle the URL and Protocol.
#     #register(project_name="tapis-agent")
    
#     qa_pairs = load_data("data_new.json")
    
#     print(f"⏳ Querying Agent for {len(qa_pairs)} benchmark questions...")
#     for i, qa in enumerate(qa_pairs):
#         print(f"[{i+1}/{len(qa_pairs)}] Question: {qa['question']}")
#         agent_answer = await run_agent(qa["question"])
#         qa["answer"] = agent_answer 

#     evaluator = QAEvaluator()
#     results = evaluator.evaluate(qa_pairs)
    
#     output_path = "benchmark_results.json"
#     with open(output_path, "w") as f:
#         json.dump(results, f, indent=4)
        
#     print(f"\n✅ Evaluation complete. Results saved to {output_path}")

# if __name__ == "__main__":
#     asyncio.run(run_benchmark())