import os
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
	parser.add_argument('-prompt_type', type=str, default='basic', help='prompt type')
	parser.add_argument('-model_type', type=str, default='hyperbolic', choices=['hyperbolic', 'together', 'openai_reasoning', 'openai_chat', 'vllm'], help='Which type of model to use')
	parser.add_argument('-model', type=str, default='deepseek-ai/DeepSeek-R1', help='Which model to use')
	parser.add_argument('-max_tokens', type=int, default=8096, help='Maximum number of tokens')
	parser.add_argument('-temperature', type=float, default=0.6, help='Sampling temperature')
	parser.add_argument('-reasoning_effort', type=str, default='low', choices=['low', 'medium', 'high'], help='Reasoning effort')
	parser.add_argument('-top_p', type=float, default=1.0, help='top what percentage of tokens to be considered') 

	return parser

def main(args):
	model = LargeLanguageModel(model_type=args.model_type, model=args.model)

	# If you have a dataset, just put the below code in a loop
	sys_prompt, prompt = get_prompt(args.prompt_type)

	response = model.predict(
		prompt=prompt,
		sys_prompt=sys_prompt,
		max_tokens=args.max_tokens,
		temperature=args.temperature,
		reasoning_effort=args.reasoning_effort,
		top_p=args.top_p,
		stop=args.stop
	)

	print("Model Response:\n\n", response)

	with open(os.path.join(args.out_dir, "output.txt"), "w") as f:
		f.write(response)


if __name__ == "__main__":
	parser = build_parser()
	args = parser.parse_args()

	args.out_dir_name = args.out_dir

	cur_time = str(datetime.datetime.now())
	disp_time = cur_time.split()[0] + "-" + cur_time.split()[1].split(".")[0]

	if args.run_name == "default":
		args.run_name = args.model + "_" + disp_time + "_" + str(random.randint(0,100))

	args.run_name = args.run_name.replace("/", "-")

	args.out_dir = os.path.join(args.out_dir, args.run_name)

	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)

	openai.api_key = os.getenv("OPENAI_API_KEY")
	together.api_key = os.getenv("TOGETHER_API_KEY")

	main(args)