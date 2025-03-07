import os
import pandas as pd 
from tqdm import tqdm
import argparse
import datetime
import random
import openai
import together
from models import LargeLanguageModel
from prompts import get_prompt

def build_parser():
	parser = argparse.ArgumentParser(description='Generate')

	parser.add_argument('-run_name', type=str, default='default', help='run name for logs')
	parser.add_argument('-out_dir', type=str, default='outputs/', help='Output Directory')
	parser.add_argument('-stop', type=list, default=[], help='When to stop generation')
	parser.add_argument('-input_directory', type=str)
	parser.add_argument('-system_prompt', type=str, default='You are a helpful and harmless assistant.', help='System prompt')
	parser.add_argument('-question_frame', type=str, default='Which of the two sentences below is grammatical, and which is ungrammatical?\nSentence A: {0}\nSentence B: {1}', help='Question frame')
	parser.add_argument('-prompt_type', type=str, default='basic', help='prompt type')
	parser.add_argument('-model_type', type=str, default='together', choices=['hyperbolic', 'together', 'openai_reasoning', 'openai_chat', 'vllm'], help='Which type of model to use')
	parser.add_argument('-model', type=str, default='deepseek-ai/DeepSeek-R1', help='Which model to use')
	parser.add_argument('-max_tokens', type=int, default=8096, help='Maximum number of tokens')
	parser.add_argument('-temperature', type=float, default=0.6, help='Sampling temperature')
	parser.add_argument('-reasoning_effort', type=str, default='low', choices=['low', 'medium', 'high'], help='Reasoning effort')
	parser.add_argument('-top_p', type=float, default=1.0, help='top what percentage of tokens to be considered') 

	return parser

def main(args):
	model = LargeLanguageModel(model_type=args.model_type, model=args.model)
	data = pd.read_csv(args.input_directory)
	order_1_responses = []
	order_2_responses = []
	for i in tqdm(range(len(data))):
		row = data.iloc[i]
		# If you have a dataset, just put the below code in a loop
		sentence_grammatical = row['sentence_grammatical']
		sentence_ungrammatical = row['sentence_ungrammatical']
		prompt_order_1 = args.question_frame.format(sentence_grammatical, sentence_ungrammatical)
		prompt_order_2 = args.question_frame.format(sentence_ungrammatical, sentence_grammatical)
		#
		response_order_1 = model.predict(
			prompt=prompt_order_1,
			sys_prompt=args.system_prompt,
			max_tokens=args.max_tokens,
			temperature=args.temperature,
			reasoning_effort=args.reasoning_effort,
			top_p=args.top_p,
			stop=args.stop
		)
		order_1_responses.append(response_order_1)
		#
		response_order_2 = model.predict(
			prompt=prompt_order_2,
			sys_prompt=args.system_prompt,
			max_tokens=args.max_tokens,
			temperature=args.temperature,
			reasoning_effort=args.reasoning_effort,
			top_p=args.top_p,
			stop=args.stop
		)
		order_2_responses.append(response_order_2)
	#
	output_df = data 
	output_df['order_1_response'] = order_1_responses
	output_df['order_2_response'] = order_2_responses
	# Melt the df so that we instead have the column "order" with values 1 and 2, and the column "response" with the corresponding response
	id_variables = [column for column in output_df.columns if column not in ['order_1_response', 'order_2_response']]
	output_df = output_df.melt(id_vars=id_variables, value_vars=['order_1_response', 'order_2_response'], var_name='order', value_name='response')
	output_df = output_df.sort_values(by=['n_attractors', 'source_idx', 'order']).reset_index(drop=True)
	#
	output_df.to_csv(os.path.join(args.out_dir, f"{args.run_name}.csv"), index=False)


if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()

	args.out_dir_name = args.out_dir

	cur_time = str(datetime.datetime.now())
	disp_time = cur_time.split()[0] + "-" + cur_time.split()[1].split(".")[0]

	if args.run_name == "default":
		args.run_name = args.model + "_" + disp_time + "_" + str(random.randint(0,100))

	args.run_name = args.run_name.replace("/", "-")

	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)

	openai.api_key = os.getenv("OPENAI_API_KEY")
	together.api_key = os.getenv("TOGETHER_API_KEY")

	main(args)
