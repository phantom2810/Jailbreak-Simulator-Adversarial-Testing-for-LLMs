
# Jailbreaking Starter: Adversarial Robustness for Language Models

This project implements an adversarial testing framework for evaluating the robustness of large language models (LLMs). It utilizes methods like Monte Carlo Tree Search (MCTS) with Double Progressive Widening (DPW) and adversarial suffix generation to test and optimize prompts against predefined ethical and moderation constraints.

## Features

- **Adversarial Testing**: Evaluate LLM responses against adversarially crafted prompts.
- **Monte Carlo Tree Search (MCTS)**: Includes a DPW solver for efficient exploration of adversarial space.
- **Support for Multiple Models**: Works with models such as GPT, Vicuna, and LLaMa.
- **Moderation API**: Integrates OpenAI's Moderation API for content filtering.
- **Visualization**: Includes tools for loss tracking and visualization.

## Code Structure

- **`KOV.py`**: The main script managing the adversarial testing process and framework setup.
- **`WhiteBox.py`**: Implements the white-box optimization methods, including token sampling and adversarial suffix generation.
- **`DPW_Solver.py`**: Contains the DPW-based MCTS solver for optimizing adversarial search.
- **`utils.py`**: Utility functions for model loading, prompt generation, moderation checks, and loss visualization.

## Requirements

- Python 3.8 or higher
- PyTorch
- Transformers
- OpenAI API (requires `OPENAI_API_KEY` environment variable)
- Additional dependencies listed in the `requirements.txt` file.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/jailbreaking-starter.git
   cd jailbreaking-starter
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your-api-key
   ```

## Usage

### Running the Framework

1. Configure the input parameters and models in `KOV.py`.
2. Run the main script:
   ```bash
   python KOV.py
   ```

### Example Output
The framework evaluates adversarial prompts and generates scores for LLM responses:
- Moderation category scores
- Adversarial success rates
- Loss and perplexity metrics

## Project Files

- **`KOV.py`**: Main script for orchestrating the adversarial testing process.
- **`WhiteBox.py`**: Implements the white-box Markov Decision Process (MDP) for adversarial prompt generation.
- **`DPW_Solver.py`**: Provides MCTS logic with progressive widening for state-action exploration.
- **`utils.py`**: Helper functions for prompt processing, scoring, and response generation.

## Contributing

Contributions are welcome! If you encounter issues or have ideas for improvement, feel free to open an issue or submit a pull request.
