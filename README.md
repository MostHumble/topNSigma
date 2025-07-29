# Top-N-Sigma

## Abstract

Large language models (LLMs) rely heavily on sampling methods to generate diverse and highquality text. While existing sampling methods like top-p and min-p have identified the detrimental effects of low-probability tails in LLMs’ outputs, they still fail to effectively distinguish between diversity and noise. This limitation stems from their reliance on probability-based metrics that are inherently sensitive to temperature scaling. Through empirical and theoretical analysis, we make two key discoveries: 
- the pre-softmax logits exhibit a clear statistical separation between informative tokens and noise
- we prove the mathematical equivalence of min-p and top-(1-p) under uniform distribution over logits. 
These findings motivate the design of top-nσ, a novel sampling method that identifies informative tokens by eliminating noise directly in logit space. Unlike existing methods that become unstable at high temperatures, top-nσ achieves temperature-invariant token selection while preserving output diversity. Extensive experiments across reasoning and creative writing tasks demonstrate that our method consistently outperforms existing approaches, with particularly significant improvements in high-temperature settings.

## Implemenation:

```py
# Calculate M (max logit) and sigma (standard deviation of logits) for each sequence in the batch
max_logit, _ = torch.max(scores, dim=-1, keepdim=True)
std_logit = torch.std(scores, dim=-1, keepdim=True)

# Calculate the filtering threshold for each sequence
threshold = max_logit - self.n * std_logit

# Create a boolean mask for tokens to be removed
tokens_to_remove = scores < threshold

# Apply the filter
scores_processed = scores.masked_fill(tokens_to_remove, self.filter_value)
return scores_processed
```

## Reference

```bibtex
@inproceedings{tang2025top,
  title={Top-n𝜎: Eliminating Noise in Logit Space for Robust Token Sampling of LLM},
  author={Tang, Chenxia and Liu, Jianchun and Xu, Hongli and Huang, Liusheng},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={10758--10774},
  year={2025}
}
```
