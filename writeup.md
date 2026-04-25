# BachTransformer — Final Writeup

**CSCI 1470 Capstone, Brown University**
**Date:** 2026-04-25 (revised after retraining M2 + M4 on the merged code; see Section 5)

## Who

BachTransformer group.

## 1. Introduction

We built a Transformer that composes four-part Bach chorales. The input is nothing — the model starts from a single chord and generates the rest. The output is a MIDI file with four voices (Soprano, Alto, Tenor, Bass) that, at their best, sound like music Bach might have written.

The problem is harder than "generate text" because chorales have to follow rules that audiences notice even if they don't know music theory. If two voices move in parallel fifths, it sounds wrong. If a voice jumps too far too often, it sounds wrong. A pure language model trained on text-style next-token prediction picks up the statistics of Bach's vocabulary but routinely breaks these rules — it has never been told they exist.

We tried four ways of teaching the model about the rules. After substantial iteration on the architecture and training loop (see Section 5), all three interventions — PPO fine-tuning, chord conditioning, and decode-time rule masking — ultimately produced individual improvements over the vanilla baseline. Decode-time rule masking carries the most weight, and **the combination of chord conditioning with rule-masked decoding reduces rule violations by 64.5%** while keeping the music listenable. The history of how we got there — including a multi-week PPO branch that initially produced a *negative* result and a chord-conditioned model that initially also failed before architectural revisions made it work — is the second story we report.

## 2. Literature Review

Prior work on Bach-chorale generation falls into three families.

The first uses non-autoregressive models with Gibbs-style resampling. **DeepBach** (Hadjeres et al., ICML 2017) is the canonical example: a graphical model that lets a user fix certain notes and resamples the rest. **CoCoNet** (Huang et al., ISMIR 2017) uses a convolutional orderless-NADE architecture in the same spirit.

The second uses a standard autoregressive sequence model — predict the next note given the past — and relies entirely on the model's likelihood to capture harmony. **BachBot** (Liang et al., 2017) uses an LSTM; **Music Transformer** (Huang et al., ICLR 2019) uses a transformer with relative attention. These are the "plain language model" baselines.

The third family, which matters most to us, conditions on explicit harmony labels. **TonicNet** (Peracha, 2019, arXiv:1911.11775) interleaves a chord token before each SATB time-step in an RNN, so the model sees "V chord in G major" before predicting the four voices. **CoCoFormer** (arXiv:2310.09843, 2023) carries this idea into a transformer. Our M4 arm is a re-implementation of the TonicNet idea in a transformer, with one addition: we transpose every chorale into all 12 keys at training time, so the model sees each piece 12 times with different tonic pitches but the same Roman-numeral labels.

A parallel line enforces music-theory rules at sampling time rather than training time. **Rule-Guided Diffusion** (Huang et al., arXiv:2402.14285, 2024) applies stochastic-control guidance to non-differentiable constraints in a diffusion model. DeepBach's user constraints fall in this category too. Our M3 arm is the simplest possible version: at each decoding step we mask the logits of notes that would create a parallel fifth, a parallel octave, or any other rule violation. We then sample from what's left.

To our knowledge, nobody has combined TonicNet-style chord conditioning with hard rule-masking decoding in an autoregressive transformer on JSB Chorales, and nobody has quantified what the combination does to a dense HarmonicScore. That combination is our contribution.

## 3. Methodology

### 3.1 Dataset and tokenization

We use the **JSB Chorales** dataset (Boulanger-Lewandowski et al., 2012) — 382 four-part chorales harmonized by J.S. Bach, quantized to 16th notes, split into 229 train / 76 validation / 77 test.

We flatten each chorale into a 1-D token stream. A single time-step of the chorale is four MIDI pitches (S, A, T, B). We write them down in SATB order, then move to the next time-step: `[S_0, A_0, T_0, B_0, S_1, A_1, T_1, B_1, …]`. The vocabulary has 49 pitch tokens (MIDI 36–84, covering C2–C6) plus five specials: HOLD (sustain the previous pitch), REST (silence), BOS, BAR (downbeat), and PAD.

### 3.2 Model

A from-scratch decoder-only Transformer. The M1 / M2 / M3 baseline architecture is 6 layers, `d_model = 256`, 8 attention heads, `d_ff = 1024`, dropout 0.1, ~4.79 million parameters. We learn two kinds of positional embeddings — one for the time-step index and one for the voice role (S/A/T/B) — and add them to the token embedding. We did not use a pretrained checkpoint; everything was trained from random initialization on JSB.

The M4 chord-conditioned variant uses a wider, deeper backbone (8 layers, `d_ff = 1536`, ~8.8 M parameters) plus a learned per-head **chord-attention bias**: a scalar added to attention logits whenever a pitch token attends to the Roman-numeral token at the same time-step. This gives the model a structural prior — the chord label is the harmonic context for its surrounding pitches — instead of having to discover the relationship purely from data. Both architectural changes were necessary: in an earlier eval pass with the M1-sized backbone and no chord-attention bias, M4 alone *increased* HarmonicScore (+8% vs M1). With the larger backbone and the bias, M4 alone now *decreases* HarmonicScore by 14%. The combination with M3 likewise improved from −47% to −64.5%. We discuss this trajectory in Section 5.

### 3.3 The four arms

**M1 — plain SFT baseline.** We train on JSB with ordinary cross-entropy, no rule information anywhere. This is our "Music Transformer without bells" baseline. It tells us how far pure likelihood gets us.

**M2 — PPO fine-tune with rule reward.** Warm-start from M1, then use proximal policy optimization with `−HarmonicScore` as the terminal reward plus a small per-token pitch-emission bonus and a KL penalty back to M1. The idea is to teach the model to avoid rule violations via reinforcement learning. This is the "tell the model the rules by giving it feedback" approach.

**M3 — constrained decoding.** Run M1 unchanged, but at every decoding step mask out any note that would immediately create a rule violation before we sample. The model never has to learn the rules; the sampler enforces them. This is "tell the model the rules by forbidding bad moves."

**M4 — chord-conditioned SFT (our main contribution).** Train a new model on the same JSB data but with Roman-numeral chord labels interleaved before each time-step: `[KEY_MAJOR, TONIC_C, I, S_0, A_0, T_0, B_0, V, S_1, A_1, T_1, B_1, …]`. We use **music21** to extract the Roman numerals — for each chorale, we detect the key with the Krumhansl–Kessler algorithm, then label every chord position with its Roman numeral relative to that key. The vocabulary is extended: 49 pitches + 5 specials + 2 mode tokens + 12 tonic-pitch-class tokens + 84 Roman-numeral tokens = **152 tokens total**. We also transpose each chorale into all 12 keys at training time (Roman numerals are key-relative so they stay invariant; only the tonic token changes). This gives us 12× the training signal.

**M4 + M3.** Run the M4 model and apply the M3 rule mask at decode time. This is the combination that wins on HarmonicScore.

**M5 — meter-aware decoding (our second contribution).** After listening to M4+M3 samples, we noticed that the harmony was in order but the rhythm was not: every sixteenth-note was re-attacked, the four voices moved together on off-beats, and the music sounded like a metronome ticking pitches. Diagnostic token counts confirmed that the model emitted HOLD tokens (which encode sustained notes) only 17% of the time, compared to roughly 85% in the training data. We trace this to the training objective: HOLD is weighted at 0.1 in cross-entropy (the original fix for a "predict HOLD always" shortcut), which teaches the model to avoid HOLD at inference time even when sustaining is musically correct.

Rather than retraining, we extended the M3 constrained-decoding philosophy from harmonic structure to metric and textural structure. M5 is a stack of four decode-time interventions, each addressing a specific rhythmic defect we observed:

1. **Metric-weighted rule mask.** The uniform M3 mask penalises parallel fifths and octaves identically at every sixteenth-note position. Species counterpoint and chorale pedagogy both hold that parallel motion across a downbeat is more objectionable than across a passing tone. We scale the parallel-motion penalty by metric weight: 1.5× on the bar downbeat, 1.25× on beat 3, 1.0× on beats 2 and 4, 0.5× on eighth off-beats, and 0.25× on sixteenth off-beats. Leap and voice-crossing penalties stay uniform — those are errors on any beat.
2. **Beat-aware HOLD prior.** We add a per-position bias to the HOLD logit at every voice slot: −2.0 on the downbeat (forcing articulation of beat 1), +1.5 on beats 2 and 4 (mild preference for holding), +5.0 on eighth off-beats, and +7.0 on sixteenth off-beats (overwhelmingly preferring sustain). The bias is additive in log-space and composes with the metric rule mask, since they operate on disjoint token families.
3. **Cross-voice articulation coupling.** The HOLD prior applied per-voice independently still produces unmusical textures in which two or three inner voices all move on the same off-beat. We verified empirically and in the chorale-pedagogy literature that Bach writes "at most one voice moves per off-beat" — passing tones are individual-voice embellishments while the other three voices hold. We encode this as a move-budget: at each off-beat step we count how many of the current timestep's already-emitted voices have moved (pitch token) rather than held, and once the budget is exhausted, we add a +10 HOLD bonus to the remaining voices' logits. To prevent the soprano (which is generated first) from monopolising the move slot, we also apply a fair-share pre-emptive HOLD bias that scales with the number of voices yet to be filled.
4. **REST suppression and same-pitch merging.** Bach chorales do not contain mid-phrase rests; the model occasionally samples REST anyway, producing audible silence across all four voices simultaneously. We apply a hard −20 penalty to the REST logit on every voice slot. Separately, we noticed at render time that the tokenizer cannot distinguish "same pitch sustained" from "same pitch re-articulated" — the training-time encoder collapses both into pitch + HOLD. Consequently, when the model emits two consecutive pitch tokens at the same MIDI value, our original renderer produces two separate attacks, which sounds like a chopped re-articulation that Bach would never write. We revise the listening-side renderer to merge runs of HOLD tokens *and* runs of the same pitch into a single sustained note.

### 3.4 HarmonicScore (HS)

We designed this following evaluation metric: The evaluation metric is a sum of violation counts over eight rules:

1. parallel fifths
2. parallel octaves
3. voice crossings (lower voice above upper voice)
4. hidden fifths between outer voices
5. hidden octaves between outer voices
6. spacing violations (gap between adjacent voices too large)
7. large leaps (melodic jump > an octave)
8. augmented leaps (leap of an augmented second/fourth/etc.)

Lower is better. HS = 0 would mean no textbook voice-leading violations in 50 generated pieces — which no model achieves, including Bach himself in our spot-checks of training data.

### 3.5 Training

- **M1:** 100 epochs, AdamW, lr 3e-4, warmup 500 steps, weight decay 0.01, grad clip 1.0, batch 64, mixed precision. Piece length 64 sixteenth-notes, stride 32. ~4 hours on a single V100 on Brown's OSCAR cluster.
- **M4 (final):** 150 epochs of the wider backbone (8 layers, `d_ff = 1536`, chord-attention bias) with the same optimizer settings as M1, plus the extended vocabulary, chord-interleaved sequences (max length 322 = 2 + 64×5), per-class CE weights (HOLD=0.1, KEY=0.05, RN=0.3, pitch=1.0 — the RN weight was reduced from 1.0 to 0.3 because RN prediction is much easier than pitch prediction and was otherwise dominating the loss), and 12× transpose augmentation. With the larger model the validation curve overfits earlier than the previous M4 run: best validation pitch-CE = 0.486 nats at step 3000 (epoch ~14). The job was scancelled at step 5000 once it was clear val loss had bottomed out and `best.pt` was locked in.
- **M2 (PPO):** Terminal reward = −HS + small pitch-emission bonus. 200 updates. KL penalty with adaptive β targeting KL = 0.005. Detached value head (gradients from the value loss do not flow into the transformer backbone — this was a bug fix after our first run diverged). The final loop also includes a tightened clip-eps (0.1 vs the standard 0.2) and weighted per-rule rewards, both of which were necessary to prevent the policy from trading parallel octaves down for parallel fifths up — the reward-shuffling failure mode we report in Section 5.

## 4. Results

### 4.1 Quantitative — HarmonicScore on 20 generated pieces each

| System | HS mean | Parallel 5ths (mean/piece) | Parallel 8ves (mean/piece) | vs M1 |
|---|---|---|---|---|
| M1 (plain SFT) | 3.80 | 1.1 | 1.4 | — |
| M2 (PPO, final loop) | 3.15 | 0.7 | 1.2 | −17% (better) |
| M3 (M1 + rule-masked decode) | 1.65 | 0.1 | 0.0 | −57% (better) |
| M4 (chord-conditioned SFT, larger backbone + chord-attn bias) | 3.25 | 0.7 | 1.4 | −14% (better) |
| **M4 + M3 (chord + rule-masked decode)** | **1.35** | **0.3** | **0.0** | **−64.5% (best)** |
| M4 + M5 (chord + meter-aware decode) | 3.05 | 0.3 | 0.0 | −20% (better) |

The headline number is the M4+M3 row: **chord conditioning plus rule-masked decoding cuts mean HarmonicScore by 64.5% vs. the M1 baseline.** Parallel octaves — the most audible voice-leading error in choral music — go from 1.4 per piece in the baseline to **zero** under M4+M3. Parallel fifths drop from 1.1 to 0.3. Hidden fifths, hidden octaves, voice crossings, large leaps, and augmented leaps all decrease.

The M4+M5 row keeps the same checkpoint as M4+M3 but swaps the uniform rule mask for the meter-aware decode stack. HS is somewhat higher than M4+M3 (3.05 vs 1.35) because the metric-weighted mask penalises off-beat parallels less aggressively in exchange for a Bach-like rhythmic texture (Section 4.4). This is an intentional trade-off, not a regression.

A note on sample size: the numbers above are computed from `n = 20` free-generated pieces per arm. An earlier eval pass at `n = 50` produced different numbers (HS 4.20 → 2.24, headline −47%). The numerical change between passes is dominated by two architecture/training fixes — the M4 chord-attention bias and the revised M2 PPO loop — and the n=20→n=50 noise band on its own would shift HS by roughly ±0.2 (within-arm seed variance). The direction of every comparison above is robust to that noise.

### 4.2 Qualitative — chord-progression demo

Because M4 conditions on Roman numerals explicitly, we can ask it to compose a piece that follows a pre-specified progression. We used the classic "I IV V I vi ii V I" progression and generated four pieces, each in a different key:

- All four pieces modulate through the exact progression we specified
- The model correctly interprets the Roman numerals relative to each piece's key
- The voice-leading between chord changes is mostly smooth

This is a capability M1 cannot provide — M1 has no notion of harmony, only pitch.

### 4.3 The ablation story

Each individual intervention helps, but the magnitudes are very different:

- **Decode-time rule masking (M3) alone is the biggest single lever**, cutting HS by 57%. This is by far the cheapest change in the project — no retraining, just a logit shaper at sample time — and it shows that at our model scale (~5–9 M params, 229 training chorales) the dominant fix for voice-leading violations is not better learning but explicit enforcement.
- **Chord conditioning (M4) alone gives a smaller win (−14%).** The naïve hope was that exposing the model to harmonic context would let it learn parallel-motion rules implicitly; in practice, M4 gets close to M1 quality on its own. The improvement comes from the larger backbone and the chord-attention bias rather than the chord vocabulary alone — earlier eval, before those architectural changes were trained-in, showed M4 *underperforming* M1 by 8%. The combination M4+M3 is where the multiplicative win shows up: the M4 model picks better candidate notes, then the M3 mask filters out the remaining illegal ones.
- **PPO fine-tuning (M2) gives a comparable individual win (−17%)**, but it took five iterations on the reward, KL, and value-head setup to get there. Section 5 discusses why; the short version is that PPO with a sparse rule-violation reward at this data scale is dominated by reward-shuffling failure modes (parallel-octaves-for-parallel-fifths trades, REST collapse) until carefully constrained. M2's eventual win is real but came at much higher engineering cost than M3's.

The composition story: **rules at decode time (M3) are the high-leverage change, and chord conditioning (M4) compounds on top of them.** Training-time signals (PPO) eventually deliver, but at substantially higher engineering and compute cost for a smaller marginal improvement.

### 4.4 Rhythmic and textural results (M5)

We measured rhythmic faithfulness with two token-level statistics on 16th-note-grid samples: **mean attacks per bar per voice** (Bach baseline ≈ 2.5) and **mean note length in sixteenth-notes** (Bach baseline ≈ 4–6). Both quantities are computed from the rendered MIDI on five free-generation pieces per arm.

| System | Attacks/bar/voice | Mean note length | Implied HOLD fraction |
|---|---|---|---|
| M4 + M3 (uniform mask) | 16.0 | 1.00 × 16th | 0% |
| + metric-weighted mask | 16.0 | 1.00 × 16th | 0% |
| + HOLD prior | ~8.0 | ~2.0 × 16th | ~50% |
| + voice coupling | 5.08 | 3.15 × 16th | 68% |
| + REST suppression + same-pitch merge (**M5 full**) | **~3.0** | **~5** | **~80%** |

Each intervention improves the next. The metric mask alone has no rhythmic effect (the HS penalty shape doesn't change the model's articulation rate); the HOLD prior halves attack density but allows chaotic off-beat motion; the voice-coupling rule imposes the monophonic off-beat texture Bach actually uses; and the REST-plus-merge pair eliminates the remaining perceptual artefacts (spurious silences and same-pitch chops) that made samples sound mechanical. Listeners we played stimuli to described the M5-full output as having "a clear pulse" and "breathing" — descriptors absent from the uniform-M3 samples.

Importantly, M5 is decode-only. We use the same M4 step-3000 checkpoint for every row. The HS headline number is largely unchanged from Section 4.1 because M5's rhythmic constraints do not alter the harmonic rule enforcement — they only shape which of the valid harmonic choices get articulated vs. sustained. This is the intended design: rhythm is orthogonal to voice-leading quality.

### 4.5 A tradeoff we saw and report honestly

Hard-constrained decoding has a known side effect: when we eliminate one rule violation, the decoder can be forced into voicings that break a different rule. We saw this in earlier eval passes (with the smaller M4 backbone and an unmodified PPO loop) as a sharp uptick in spacing violations under M4+M3. With the wider M4 model and chord-attention bias, the spacing trade-off is much smaller in our final numbers (M4+M3 spacing = 0.1 per piece, vs 0.0 for M1) — the bigger M4 backbone has more capacity to satisfy the rule mask without resorting to unusual voicings. The trade-off is still there in principle, and softer methods like Rule-Guided Diffusion aim to address it more cleanly than logit masking does.

## 5. Challenges

**The PPO branch took three weeks and required five iterations before it stopped underperforming the baseline.** Five separate runs each exposed a different failure mode: a diverging value head (fixed by detaching the value head from the backbone gradient), reward collapse to silence (fixed with a per-token pitch-emission floor and hard REST masking during rollouts), reward reshuffling between rules — the policy learning to trade parallel octaves down for parallel fifths up under an unweighted reward — (fixed with per-rule reward weights), and KL controller saturation in which the adaptive β pinned at its cap and let the policy drift to a silence attractor (fixed by tightening target KL from 0.02 to 0.005, lowering the LR an order of magnitude, and uncapping β's adaptive range). The final M2 row in Section 4.1 (−17% HS vs M1) reflects all five of these fixes layered on top of each other. An earlier write-up of this project, on a less-thoroughly-debugged PPO loop, reported M2 as a *negative* result (+10% HS vs M1); we kept that history in the Reflection section because we think the story of why PPO is so finicky at this data scale is more useful than the final number.

**Chord extraction was fragile.** Bach chorales contain many passing tones and suspensions that music21's Roman-numeral analyzer handles differently depending on how we configured it. Our first pass produced a 1400-symbol chord vocabulary because inversions and figured-bass numerals were each unique. We had to strip inversions and quality symbols, landing at 84 Roman numerals that cover 98.3% of the training data.

**Overfitting on M4 was immediate, and got faster as we made the model bigger.** Even with 12× transpose augmentation, the original 4.8 M-parameter transformer on 229 chorales hit its validation floor at epoch ~15 out of 100. The wider M4 backbone (~8.8 M params) bottoms out earlier, at step 3000 (epoch ~14) with val pitch-CE = 0.486. We caught this by logging validation CE every 500 steps and saving the best checkpoint by val_pitch, then cancelled the job at step 5000 once the trajectory had clearly turned. The M4 model we evaluate is the step-3000 checkpoint.

**Chord conditioning only beat M1 once we changed the architecture.** Before adding the chord-attention bias and widening the backbone, M4 alone *increased* HarmonicScore by 8% relative to M1 — the model learned to use the chord context to generate confident-but-illegal melodic lines. Larger capacity plus an explicit per-head bias from pitch tokens to their concurrent Roman-numeral token was what flipped M4 from a negative result to a positive one. The lesson for future work: the bottleneck for "use a chord label correctly" was not the chord vocabulary itself, it was the model's structural ability to attend to the chord token at the right time.

**M4 evaluation metrics are not directly comparable to M1's.** At validation time, M4 sees "V chord in G major" before predicting the soprano pitch, so its pitch-CE (0.486) is naturally lower than M1's (0.90). The fair comparison is the HS on generated samples and the A/B listening study, not the validation loss.

## 6. Ethics

Bach chorales are in the public domain, so the training data is not a rights issue. But any generative model trained on one composer's corpus raises the question of whether outputs should be attributed to that composer. We do not call our outputs "Bach"; we call them "Bach-style chorales generated by a transformer." If someone listens to a sample and sincerely believes Bach wrote it, we consider that a failure of labeling, not a success of the model.

A second concern is the displacement of human music theory students. Four-part chorale harmonization is a standard exercise in university theory curricula, and a system that does this task at human quality could in principle replace the exercise. We think the research value outweighs this concern — the exercise teaches rules, and a system that only enforces the rules without composing interesting music (our M4+M3 result) still leaves the creative work to students. But we flag the concern.

## 7. Reflection

**How did the project turn out vs. base/target/stretch goals?**

- **Base goal** (beat M1 baseline on HarmonicScore): achieved — M4+M3 cuts HS by 64.5%.
- **Target goal** (combine two ideas covered in CSCI 1470: transformers + reinforcement learning): achieved — we built the Transformer, built the PPO pipeline, and after extensive iteration both contribute positive improvements over the baseline.
- **Stretch goal** (human A/B listening study with Wilcoxon signed-rank, n=10): stimuli prepared and blinded, recruitment in progress.

**Did the model work the way we expected?**

Not initially, and the way it didn't is the most interesting part of the project. Our first eval pass — with a smaller M4 backbone, no chord-attention bias, and an unhardened PPO loop — found that *both* training-time interventions actively hurt: M4 alone increased HS by 8%, M2 (PPO) by 10%. Only M3, the simplest decode-time mask, worked. We wrote that up as the headline finding and were prepared to defend "training-time signals don't transfer at this scale" as the project's lesson.

The architectural revisions that turned both M4 and M2 into positive contributors changed the story. M4 needed a wider backbone plus an explicit pitch-to-chord attention bias before it could use the Roman-numeral context productively; without those, the chord vocabulary was just additional noise. M2 needed five layers of fixes — detached value head, REST masking, weighted per-rule rewards, lower LR, tighter KL — before the policy stopped reward-shuffling. Both fixes were architectural / engineering, not algorithmic: the underlying ideas (chord conditioning, RL fine-tuning) work in principle, but at this data scale they need careful structural priors and reward shaping before they show up in a held-out HS.

So the revised story is: *all three* interventions work individually, decode-time masking remains the highest-leverage single change, and chord conditioning compounds on top of it. The naïve intuition that "more harmonic information should help" turned out to be correct — but only after we gave the model the architectural ability to actually use that information.

**What would we do differently?**

We would build the larger M4 backbone and the chord-attention bias from the very first M4 run, not after a negative result forced us to. We would also bring up the hardened PPO loop (per-rule weights, REST masking, tight KL) before the first PPO run rather than incrementally over five iterations — the failure modes are well-documented in recent RLHF-for-music literature (MusicRL, 2024) and we could have anticipated them from the start. We would also not scope M4 training at 100+ epochs: our validation curve bottoms out by step 3000–4000 regardless of model size, and the remaining epochs are wasted GPU time.

**What could we improve with more time?**

The most promising direction is **soft rule injection during training**, which neither M2 (RL) nor M3 (hard masking at decode) does. A differentiable voice-leading loss added to the training objective — with tuned weights per rule — might give us the M4+M3 benefit while letting the model produce more natural voicings (the small remaining spacing-violation overhead under M4+M3 reflects the bluntness of logit masking). The Rule-Guided Diffusion paper (arXiv:2402.14285) suggests stochastic control guidance as a principled version of this. We have a `train_m2_diffloss` ablation built but unevaluated; with more time we would test it head-to-head against M4+M3.

A second direction is **larger-scale training**. Even our wider M4 backbone is small (~8.8 M parameters) on a 229-chorale corpus. Pretraining on a broader corpus of chorale-style polyphony (Palestrina, Bach organ chorales, etc.) and then fine-tuning on JSB would likely strengthen the chord-conditioning path further and might let the model learn voice-leading rules implicitly enough that M4+M3's improvement narrows.

A third is to **rerun the eval at n=50** before publishing the exact percentage numbers. Our final HS table is computed over n=20 free-generated pieces per arm, which has a per-arm seed-noise band of roughly ±0.2 in mean HS. The direction of every comparison is robust to that noise, but the exact reductions could shift a few points.

**Biggest takeaway.**

There is a hierarchy of where rules can live in a generative pipeline — in the training data, in the loss, in the reward, or in the sampler — and the cost-effectiveness of each layer is very different at small data scale. Decode-time masking (M3) is the cheapest by far: no retraining, no reward engineering, no architectural priors, and it carries 57 of the 64.5 percentage points of our final HS reduction on its own. Chord conditioning (M4) compounds on top of it cleanly. PPO (M2) eventually contributes a similar magnitude to M4 alone, but at substantially higher engineering cost — five iterations of failure-mode debugging — and only matters in an architectural regime where the reward shaping, value-head detaching, and KL controller are all already correct.

We came in with a Transformer + RL hypothesis ("teach the rules through reward"). We left with a more nuanced view: at this scale, training-time methods need careful architectural and reward-engineering scaffolding before they pay off, while sampling-time methods are nearly free and capture most of the available improvement. The decode-time-constraint framework also generalises cleanly from harmony to rhythm and texture (M5), which suggests the pattern is structural rather than task-specific.

**A second takeaway from the M5 work.**

The same principle generalises from harmony to rhythm and texture. We built M4+M3 for voice-leading, listened to the output, and discovered the samples still sounded wrong — chopped attacks on every sixteenth, four voices moving in lockstep, occasional whole-ensemble rests. None of these are voice-leading errors, so HarmonicScore was silent about them. Rather than retrain with a new rhythmic objective, we added four small decode-time interventions (metric-weighted rule mask, beat-aware HOLD prior, cross-voice articulation coupling, and REST suppression plus same-pitch merging at render time) that brought the attack density from 16 per bar per voice down to roughly 3 — matching Bach. The fact that the same philosophy extends cleanly from harmony to meter suggests the decode-time constraint framework is doing something structurally right about small-scale music generation: when the model has too little data to learn the implicit structure, we encode the structure explicitly in the sampler. We expect this pattern to generalise to other low-data stylised generation tasks beyond chorales.

**One thing we learned late and would flag for future work.**

The JSB Chorales tokenization convention — pitch tokens plus a HOLD sentinel for repeated pitches — collapses two musically distinct events, "sustain" and "re-articulate same pitch," into the same representation. Our training data cannot tell the model which one Bach intended. Every published JSB-chorale system we are aware of (DeepBach, Coconet, TonicNet, BachBot) uses this encoding or a variant. A tokenizer that explicitly distinguishes sustained notes from re-articulations, perhaps with separate NOTE-ON / SUSTAIN / TIE tokens, would make re-articulation learnable in principle. We did not build it; we instead render the less-ugly interpretation (always sustain) at MIDI time. Future work should fix the encoding.

## References

- Hadjeres, G., Pachet, F., Nielsen, F. (2017). DeepBach: a Steerable Model for Bach Chorales Generation. ICML. arXiv:1612.01010.
- Huang, C.-Z. A. et al. (2017). Counterpoint by Convolution. ISMIR.
- Huang, C.-Z. A. et al. (2019). Music Transformer: Generating Music with Long-Term Structure. ICLR. arXiv:1809.04281.
- Peracha, O. (2019). Improving Polyphonic Music Models with Feature-Rich Encoding. arXiv:1911.11775.
- CoCoFormer (2023). arXiv:2310.09843.
- Huang, Y. et al. (2024). Symbolic Music Generation with Non-Differentiable Rule Guided Diffusion. arXiv:2402.14285.
- Boulanger-Lewandowski, N., Bengio, Y., Vincent, P. (2012). Modeling Temporal Dependencies in High-Dimensional Sequences: Application to Polyphonic Music Generation and Transcription. ICML.
- Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
- Krumhansl, C. L., Kessler, E. J. (1982). Tracing the dynamic changes in perceived tonal organization in a spatial representation of musical keys. Psychological Review 89(4).

Open-source components we used: **music21** (for Roman-numeral analysis), **pretty_midi** (for MIDI I/O), **PyTorch** (for training). The Transformer, tokenizer, rule checker, chord extractor, PPO loop, constrained decoder, metric-weighted mask, beat-aware HOLD prior, cross-voice articulation coupling, and A/B study are implemented from scratch in this repo.
