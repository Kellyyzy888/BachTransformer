# Poster Discussion & References

## Poster-ready discussion

Our results suggest a clear lesson about where music-theory rules are most effective in a small-data symbolic-generation pipeline. Training-time methods did help once the implementation was stabilized: the retrained chord-conditioned model (M4) improved over the vanilla baseline, and PPO fine-tuning (M2) also became a positive contributor after several reward and optimization fixes. However, the strongest single intervention was still decode-time rule enforcement. Hard masking during sampling (M3) produced the largest drop in HarmonicScore, and combining it with chord conditioning (M4+M3) gave the best overall system, reducing mean HarmonicScore from 3.80 to 1.35.

This points to an important practical takeaway. At the JSB Chorales scale, the model does not reliably internalize all of the local voice-leading constraints from likelihood training alone, even when given explicit harmonic context. In contrast, sampling-time constraints are direct, interpretable, and cheap to add. In our project, they captured most of the available gain without requiring retraining. Chord conditioning still mattered, but it worked best as a complement to rule-masked decoding rather than as a replacement for it.

There are also several limitations to our implementation. First, HarmonicScore measures local rule compliance, not full musical quality. A sample can satisfy parallel-motion and spacing constraints while still sounding mechanical, especially rhythmically. This is exactly what motivated our later M5 meter-aware decoding experiments. Second, hard constraints can be blunt: preventing one violation can push the decoder toward less natural voicings or other stylistic compromises. Third, our final retrained evaluation uses only 20 generated samples per arm, which is enough to support the directional result but not ideal for claiming highly precise percentage improvements.

Future work should focus on softer and more scalable ways to inject structure. One promising direction is differentiable rule-aware training, such as the `train_m2_diffloss` ablation we implemented but did not fully evaluate. Another is broader pretraining on chorale-style polyphony before fine-tuning on JSB, which may let the model learn more of the harmonic and textural priors implicitly. Finally, rerunning all conditions at a larger evaluation size and extending human listening studies would give a more complete picture of how well low HarmonicScore aligns with perceived musicality.

## Poster-ready references

1. Hadjeres, G., Pachet, F., and Nielsen, F. DeepBach: a Steerable Model for Bach Chorales Generation. ICML, 2017.
2. Huang, C.-Z. A., Cooijmans, T., Roberts, A., Courville, A., and Eck, D. Counterpoint by Convolution. ISMIR, 2017.
3. Huang, C.-Z. A. et al. Music Transformer: Generating Music with Long-Term Structure. ICLR, 2019.
4. Peracha, O. Improving Polyphonic Music Models with Feature-Rich Encoding. arXiv:1911.11775, 2019.
5. Huang, Y. et al. Symbolic Music Generation with Non-Differentiable Rule Guided Diffusion. arXiv:2402.14285, 2024.
6. Schulman, J. et al. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017.

## Compressed poster version

Our main lesson is that rule information was most effective at decode time. Although retrained training-time methods eventually helped, hard rule-masked decoding gave the largest single improvement, and the combined M4+M3 system achieved the best result overall. This suggests that at the JSB scale, explicit sampling-time enforcement is more reliable than expecting the model to fully learn voice-leading constraints from data alone.

The main limitations are that HarmonicScore captures local rule violations rather than complete musicality, hard masking can introduce stylistic tradeoffs, and our final evaluation uses only n = 20 samples per arm. Future work includes evaluating differentiable rule-aware losses, scaling evaluation and listening studies, and exploring broader chorale pretraining or softer guidance methods that preserve naturalness while still enforcing structure.
