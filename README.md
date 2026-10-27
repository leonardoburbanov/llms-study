# LLMs from scratch: study notebooks

<p align="center">
  <strong>The pieces inside an LLM, built by hand in NumPy and PyTorch: embeddings, tokenizers and attention.</strong>
</p>

<p align="center">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-from%20scratch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://numpy.org/"><img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"></a>
  <a href="https://cohere.com/"><img src="https://img.shields.io/badge/Cohere-embeddings-39594D?style=for-the-badge" alt="Cohere"></a>
  <a href="https://docs.astral.sh/uv/"><img src="https://img.shields.io/badge/uv-package%20manager-DE5FE9?style=for-the-badge&logo=uv&logoColor=white" alt="uv"></a>
</p>

Most examples use huge datasets. These notebooks keep the data tiny (a two-sentence corpus, a ten-car catalog) so every matrix stays small enough to print and read. The notebook text is in Spanish, and the code is plain Python.

## Notebooks

| Notebook | What it builds |
|---|---|
| [`01_vectors_and_matrixes/01_01_vectors_for_words.ipynb`](01_vectors_and_matrixes/01_01_vectors_for_words.ipynb) | First intuition: words as vectors |
| [`embeddings101.ipynb`](embeddings101.ipynb) | **Skip-gram word embeddings in NumPy**: tokenizer, vocabulary, training pairs, and the embedding and output matrices trained by hand on a two-sentence corpus |
| [`embeddings102.ipynb`](embeddings102.ipynb) | **BPE tokenizer from scratch** (merges, encode, token IDs) on a car catalog, then the Q/K/V projection weights of an attention layer in PyTorch, visualized |
| [`embeddings103.ipynb`](embeddings103.ipynb) | **A tiny transformer encoder used as a search model**: a BPE tokenizer trained with Hugging Face `tokenizers`, then attention, LayerNorm and feed-forward wired by hand, trained on title–description pairs and queried with cosine similarity |
| [`attention_01.ipynb`](attention_01.ipynb) | **Self-attention step by step**: dot-product scores → softmax weights → context vectors, then trainable Q/K/V, causal masking, dropout and multi-head attention |
| [`embeddings_w_gemma.ipynb`](embeddings_w_gemma.ipynb) | Production embeddings (Cohere): plotting product descriptions and comparing their cosine similarities |

## Run it

```bash
uv sync
uv run jupyter lab
```

`embeddings_w_gemma.ipynb` needs a `COHERE_API_KEY` in a `.env` file. The other notebooks run offline.

## Credits

`attention_01.ipynb` follows chapter 3 of Sebastian Raschka's *Build a Large Language Model (From Scratch)*.
