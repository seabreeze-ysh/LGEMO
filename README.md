# LGEMO Search

This project extends [TextAttack](https://github.com/QData/TextAttack) with a custom LGEMO adversarial text search method. The repository includes the modified TextAttack source code.

LGEMO generates adversarial examples through word substitutions. It maintains a candidate population based on attack scores and the number of modified words, while tracking previously attempted modification positions for each candidate text.

## Environment

The project was tested on AutoDL with the following environment:

| Component | Version |
| --- | --- |
| Python | 3.9 |
| PyTorch | 2.8.0 |
| CUDA (PyTorch build) | 12.8 |

We recommend using a dedicated Python environment.

## Installation

Clone this repository and navigate to its root directory:

```bash
git clone <repository-url>
cd <repository-directory>
```

Install the dependencies and the modified TextAttack framework:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

The official `textattack` package is excluded from `requirements.txt`. The modified framework is installed directly from this repository using `pip install -e .`.

If your AutoDL image already includes PyTorch, check its version before installing dependencies.

Verify that Python loads TextAttack from this repository:

```bash
python -c "import textattack; print(textattack.__file__)"
```

The output should point to the `textattack` directory in your cloned repository.

## Search Method

LGEMO performs the following steps:

1. Identify modifiable word positions, excluding stopwords.
2. Select a parent text from the candidate population.
3. Select a modification position that has not been attempted for that text.
4. Generate substitution candidates and evaluate their attack scores.
5. Update the population using dominance comparisons based on attack scores and the number of modified words.
6. After finding a successful adversarial example, continue searching for successful examples with fewer modified words.

The method prioritizes successful adversarial examples with fewer modified words. If no successful example is found, it returns the candidate with the highest attack score in the population.

## Configuration

The `LGEMO` constructor accepts the following parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `max_iters` | `1500` | Maximum number of search iterations |
| `unchanged_iters` | `700` | Maximum number of consecutive iterations without a population update |

Configure these parameters in Python:

```python
search_method = LGEMO(
    max_iters=1500,
    unchanged_iters=700,
)
```

Alternatively, set environment variables in your AutoDL terminal:

```bash
export LGEMO_MAX_ITERS=1500
export LGEMO_UNCHANGED_ITERS=700
```

Explicit constructor arguments take precedence over environment variables.

Search iterations are different from model queries: a single iteration may evaluate multiple substitution candidates.

## Running Experiments

Replace the following placeholders with your experiment entry point and arguments:

```bash
python <experiment_script.py> <experiment_arguments>
```

Prepare the target model and dataset before running experiments. Configure model downloads, dataset loading, cache locations, and output paths according to your experimental setup.

## Acknowledgments and License

This project builds on TextAttack. We thank its authors and contributors.

The original TextAttack license and copyright notices must be retained when redistributing the framework. TextAttack and third-party components remain subject to their respective licenses. See the license documentation in this repository for the terms applicable to newly added code.
