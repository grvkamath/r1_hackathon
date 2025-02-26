def get_prompt(prompt_type="basic", params=None):
	sys_prompt = ""
	prompt = ""

	if prompt_type == "basic":
		sys_prompt = "You are a helpful and harmless assistant."
		prompt = f"""Answer the following:
Which is the bigger number: 9.11 or 9.9?
"""

	elif prompt_type == "basic_params":
		sys_prompt = "You are a helpful and harmless assistant."
		prompt = f"""Answer the following:
Which is the bigger number: {params[0]} or {params[1]}?
"""
	return sys_prompt, prompt