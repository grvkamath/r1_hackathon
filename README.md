<h2 align="center">
  Team Reddy R1 Hackathon
</h2>

# Setup

Install VirtualEnv using the following (optional):

```shell
$ [sudo] pip install virtualenv
```

Create and activate your virtual environment (optional):

```shell
$ virtualenv -p python3 renv
$ source renv/bin/activate
```

Depending on your machine, you may have to do:

```shell
$ python3 -m venv renv
$ source renv/bin/activate
```

## Dependencies

- compatible with python 3
- dependencies can be installed using `r1_hackathon/requirements.txt`
- Works best with CUDA 12.5 (otherwise you may have to struggle with installation of individual libraries)

Install all the required packages:

at `r1_hackathon/:`

```shell
$ pip install -r requirements.txt
```

## API Keys

Please set environment variables (in your `~/.bashrc` file) for `OPENAI_API_KEY`, `TOGETHER_API_KEY`, `HYPERBOLIC_API_KEY`.

# Usage

At `r1_hackathon/`:
```shell
$ python single_inference.py -prompt_type basic -model_type hyperbolic -model deepseek-ai/DeepSeek-R1 -max_tokens 4096 -temperature 0.6 -reasoning_effort low -run_name r1_basic_1
```

This will create a new directory within `r1_hackathon/outputs` with the name `r1_basic_1` which will store the model generated output in a `output.txt` file.

If you have a dataset of prompts, please implement the appropriate logic as required by simply running a loop in the `main` function.