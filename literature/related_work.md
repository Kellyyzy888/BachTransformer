# Related Work — BachGPT / Bach Transformer

This file is the literature positioning for the CSCI 1470 capstone writeup. Drop
sections into the Overleaf `related_work` section as needed.

## Comparison matrix

| Work | Arch | Chord cond. | Decode-time rule enforcement | Dataset |
|---|---|---|---|---|
| DeepBach (Hadjeres et al., ICML 2017) | Graphical, pseudo-Gibbs | No | Soft (user positional constraints) | JSB |
| CoCoNet (Huang et al., ISMIR 2017) | Convnet orderless NADE | No | Soft (resampling) | JSB |
| BachBot (Liang et al., 2017) | LSTM | No | No | JSB |
| Music Transformer (Huang et al., ICLR 2019) | Transformer (relative attn) | No | No | Piano (not Bach) |
| TonicNet (Peracha, 2019, arXiv:1911.11775) | GRU/RNN | Yes — [CHORD, S, A, T, B] interleaved | No | JSB |
| CoCoFormer (2023, arXiv:2310.09843) | Transformer (feature-rich) | Yes | No | Polyphonic |
| Rule-Guided Diffusion (2024, arXiv:2402.14285) | Diffusion + SCG | No | Yes (non-differentiable rules) | Symbolic |
| Mode-Guided Tonality Injection (2025, arXiv:2512.17946) | Transformer + FiLM | Mode only (Krumhansl–Kessler) | No | Symbolic |
| **Ours (M1 / M4 / M4+M3)** | Transformer decoder-only, 4.8M | Yes (RN interleaved) | Yes (hard logit masking, 8 rules) | JSB + 12-key aug |

## One-sentence contribution

We combine TonicNet-style chord-interleaved conditioning with decode-time
hard-constrained logit masking on a 4.8M-parameter Transformer, reducing mean
HarmonicScore by 47% vs. an unconditional SFT baseline on JSB Chorales
(4.20 → 2.24, n=50). Chord conditioning alone does not improve rule-compliance;
the combined model does.

## Literature Review paragraph (drop-in)

Prior work on Bach chorale generation falls into three rough families. The
first — DeepBach (Hadjeres et al., 2017) and CoCoNet (Huang et al., 2017) —
uses graphical or orderless-NADE architectures with Gibbs-style resampling,
giving users soft positional constraints. The second — BachBot (Liang et al.,
2017) and Music Transformer (Huang et al., 2019) — treats the chorales as a
left-to-right autoregressive sequence and relies purely on likelihood to
capture harmony. The third, most directly relevant family conditions on
explicit harmonic labels: TonicNet (Peracha, 2019) interleaves a chord token
before each SATB frame in an RNN, and CoCoFormer (2023) carries this idea
into a transformer. A parallel line of work enforces music-theory rules at
sample time rather than training time: Rule-Guided Diffusion (Huang et al.,
2024) applies stochastic-control guidance to non-differentiable constraints in
a diffusion backbone, and DeepBach's interactive constraints fall in this
category as well. To our knowledge, no prior work combines TonicNet-style
chord conditioning with *hard* rule-masking decoding in an autoregressive
Transformer on JSB, nor quantifies the effect on a dense HarmonicScore
aggregating eight traditional voice-leading rules. Our M4 arm adopts the
chord-interleaved conditioning; our M3 constrained decoder enforces the eight
rules as logit masks at each voice slot; the combined M4+M3 system is the
contribution we evaluate.

## Discussion paragraph (drop-in)

Our ablation tells a sharper story than the headline number: chord
conditioning alone (M4) does not reduce HarmonicScore — it slightly increases
it (4.56 vs. M1's 4.20). It is the combination with hard decode-time masking
(M4+M3) that drives the 47% reduction. This suggests that at this model scale
(4.8M params, 229 training chorales) implicit harmonic scaffolding is not
sufficient for voice-leading correctness; explicit enforcement is required. We
also observe a characteristic tradeoff of hard-constrained decoding:
eliminating forbidden parallel motion pushes violations into the voice-spacing
rule (+10 over baseline), a known shortcoming of rule-masking approaches that
softer methods like Rule-Guided Diffusion aim to address.

## BibTeX

```bibtex
@inproceedings{hadjeres2017deepbach,
  title={{DeepBach}: a Steerable Model for {Bach} Chorales Generation},
  author={Hadjeres, Ga{\"e}tan and Pachet, Fran{\c{c}}ois and Nielsen, Frank},
  booktitle={Proceedings of the 34th International Conference on Machine Learning},
  volume={70},
  pages={1362--1371},
  year={2017},
  publisher={PMLR}
}

@article{peracha2019tonicnet,
  title={Improving Polyphonic Music Models with Feature-Rich Encoding},
  author={Peracha, Omar},
  journal={arXiv preprint arXiv:1911.11775},
  year={2019}
}

@inproceedings{huang2019music,
  title={Music {Transformer}: Generating Music with Long-Term Structure},
  author={Huang, Cheng-Zhi Anna and Vaswani, Ashish and Uszkoreit, Jakob and Shazeer, Noam and Simon, Ian and Hawthorne, Curtis and Dai, Andrew M. and Hoffman, Matthew D. and Dinculescu, Monica and Eck, Douglas},
  booktitle={International Conference on Learning Representations},
  year={2019}
}

@inproceedings{huang2017coconet,
  title={Counterpoint by Convolution},
  author={Huang, Cheng-Zhi Anna and Cooijmans, Tim and Roberts, Adam and Courville, Aaron and Eck, Douglas},
  booktitle={Proceedings of the 18th International Society for Music Information Retrieval Conference (ISMIR)},
  year={2017}
}

@article{huang2024ruleguided,
  title={Symbolic Music Generation with Non-Differentiable Rule Guided Diffusion},
  author={Huang, Yujia and others},
  journal={arXiv preprint arXiv:2402.14285},
  year={2024}
}

@article{cocoformer2023,
  title={{CoCoFormer}: A Controllable Feature-Rich Polyphonic Music Generation Method},
  journal={arXiv preprint arXiv:2310.09843},
  year={2023}
}
```

## Sources consulted

- DeepBach: https://arxiv.org/abs/1612.01010 (ICML 2017, Hadjeres/Pachet/Nielsen)
- TonicNet: https://arxiv.org/pdf/1911.11775 (Peracha 2019)
- Music Transformer: https://arxiv.org/pdf/1809.04281 (Huang et al. ICLR 2019)
- CoCoFormer: https://arxiv.org/pdf/2310.09843 (2023)
- Rule-Guided Diffusion (SCG): https://arxiv.org/html/2402.14285v4 (2024)
- Mode-Guided Tonality Injection: https://www.arxiv.org/pdf/2512.17946 (2025)
- asdfang/constraint-transformer-bach (unpublished impl.):
  https://github.com/asdfang/constraint-transformer-bach
