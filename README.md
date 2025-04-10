# Noiser: Bounded Input Perturbations for Attributing Large Language Models

This repository contains the code for the paper ["Noiser: Bounded Input Perturbations for Attributing Large Language Models"](https://arxiv.org/abs/2504.02911)

## Noiser
We propose *Noiser*, a new perturbation-based feature attribution method for LLMs that adds controlled noise to input embeddings to measure token importance. We also introduce an *answerability* metric using a judge model to evaluate if top-attributed tokens can recover the output. Experiments across 6 LLMs and 3 tasks show that Noiser outperforms existing methods in faithfulness and answerability.

1. **Rationalization Analysis** (`main.py`): Evaluates how well language models can explain their predictions using various importance scoring methods.
2. **Answerability Analysis** (`answerability.py`): Assesses the ability of language models to provide meaningful and relevant completions based on given prompts.

![image](https://github.com/qasemii/noiser-private/blob/main/img/noiser_sample.png)

## Features

- Support all feature attribution methods from [Inseq](https://github.com/inseq-team/inseq/), as well as `attention_last` and `attention_rollout` adapted from [ReAgent](https://github.com/casszhao/ReAGent).

![image](https://github.com/qasemii/noiser-private/blob/main/img/methods_comparison.png)

### Installation
Clone the repository and install dependencies:
```
git clone https://github.com/qasemii/noiser
cd noiser
pip install accelerate datasets peft inseq
```

### Faithfulness Metrics Evaluation:
```
python main.py\
      --model_name $MODEL_NAME\
      --method $METHOD\
      --dataset $DATASET\
      --n_samples -1\
      --norm $NORM
```

Command Line Arguments

- `--model_name`: Name of the language model to use (default: "Qwen/Qwen2-0.5B")
- `--dataset`: Dataset to use for evaluation (options: `Knowns`, `LongRA`, `wikitext`)
- `--output_dir`: Directory to save results (default: "results/")
- `--n_samples`: Number of samples to evaluate (default: -1 (all samples). Set a positive number to reduce the number of samples.)
- `--max_new_tokens`: Maximum number of new tokens to generate (default: 1. For `wikitext` we set 10.)
- `--norm`: Normalization method for importance scores (default: `None`. Only set for `Noiser`)
- `--mode`: Mode for importance score calculation
- `--method`: Importance scoring method to use


### Answerability Metrics Evaluation:
![image](https://github.com/qasemii/noiser-private/blob/main/img/answerability.png)

```
python answerability.py\
      --model_name $MODEL_NAME\
      --method $METHOD\
      --dataset $DATASET\
      --n_samples -1\
      --topk 50\
      --openai_api_key $TOGETHER_API_KEY
```

Command Line Arguments

- `--model_name`: Name of the language model to use (default: "Qwen/Qwen2-0.5B")
- `--dataset`: Dataset to use for evaluation (options: `Knowns`, `LongRA`, `wikitext`)
- `--output_dir`: Directory to save results (default: "results/")
- `--n_samples`: Number of samples to evaluate (default: -1 (all samples). Set a positive number to reduce the number of samples.)
- `--max_new_tokens`: Maximum number of new tokens to generate (default: 1. For `wikitext` we set 10.)
- `--norm`: Normalization method for importance scores (default: `None`. Only set for `Noiser`)
- `--mode`: Mode for importance score calculation
- `--method`: Importance scoring method to use
- `--topk`: Top-k percentage of tokens to consider (default: 50)
- `--openai_api_key`: OpenAI API key (required for answerability analysis)






## Cite if you find this helpful 😃:
```
@misc{madani2025noiserboundedinputperturbations,
      title={Noiser: Bounded Input Perturbations for Attributing Large Language Models}, 
      author={Mohammad Reza Ghasemi Madani and Aryo Pradipta Gema and Gabriele Sarti and Yu Zhao and Pasquale Minervini and Andrea Passerini},
      year={2025},
      eprint={2504.02911},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.02911}, 
}
```


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.



