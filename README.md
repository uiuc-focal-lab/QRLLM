# Quantifying Risks in Multi-turn Conversation with Large Language Models

This repository contains the implementation code for the paper *"Quantifying Risks in Multi-turn Conversation with Large Language Models"*.

## Getting Started

1. **Set up your environment**  
    1. Create and activate conda environment: `conda env create -f env.yml; conda activate qrllm`
    2. Upload your API keys to the `.env` file.

2. **Run experiments**  
   Example command:  

   ```bash
   python random_attack.py \
       --target_model_name us.meta.llama3-3-70b-instruct-v1:0 \
       --sample_strategy adaptive \
       --jailbreak_prob 0 \
       --pre_attack_data_path ./pre_attack_result/chemical_biological/chembio.json

3. **Certify**

   Run `python certify.py <attack result file path>` on the attack result file generated from the previous step. 

## Models and Strategies

For our experiments, we accessed models via AWS Bedrock. 

--target_model_name values:

- anthropic.claude-sonnet-4-20250514-v1:0

- meta.llama3-3-70b-instruct-v1:0

- mistral.mistral-large-2407-v1:0

- deepseek.r1-v1:0

- openai.gpt-oss-120b-1:0

--sample_strategy options:

- random_node

- graph_path_vanilla

- graph_path_constraint

- adaptive

--jailbreak_prob values:

- 0

- 0.2

--pre_attack_data_path values:

- pre_attack_set/cybercrime/cyber.json

- pre_attack_set/chemical_biological/chembio.json

--num_start and --num_end:

- You can specify which portion of the dataset to use with these two arguments.

- For example:

   - Chemical_Biological dataset: it has 28 data points, so the valid range is 0–28.

   - Cybercrime dataset: it has 40 data points, so the valid range is 0–40.

- This allows you to run certification on a subset of the dataset by adjusting the start and end indices.

## File Structure

- pre_attack_set/ – Contains the generated query sets for each scenario on both datasets.

- prompts/ – Prompts used in experiments.

- build_graph.py – Builds a graph on the query set.

- jailbreak.py – Generates jailbreak prompts for insertion.

- judge.py – Judges model definition.

- random_attack.py – Main script for running experiments.

## Notes

- Ensure .env is properly configured before running any scripts.

