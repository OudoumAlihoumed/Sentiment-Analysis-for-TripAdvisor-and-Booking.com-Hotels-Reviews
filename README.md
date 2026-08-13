# Can we trust deception probes once the agent is trying?

**A robustness-gap protocol, and initial bounds on Llama-3.3-70B against the published Apollo probe**

*Oudoum A. Houmed*

*Epistemic status:* Independent empirical writeup. I trust the **transfer** result (0.999 → 0.83) most. The adaptive SFT result is only a **budget bound**: both probe-aware cells were `attack_too_weak`. I did not train a new probe; I stress-tested a published one. Analysis code later; evasive LoRAs stay gated.

---

## TL;DR

Probe papers often report near-ceiling AUROC on their own domain, against agents that are not trying to dodge the probe. I asked a different question: **how much detection remains as adversary budget rises**, and **when is an AUROC drop even evasion**?

Setup: Llama-3.3-70B-Instruct, frozen Apollo residual probe (layer 22, response mean-pool), Modal `H200:2`.

- **Transfer, no attack:** 0.999 → **0.83** [0.75, 0.90] (Apollo insider-trading → my agentic episodes).
- **RL** ($b_{4}$, $\lambda=8$): AUROC 0.75, but retention **0.14** → I **withhold** $\Delta_{\mathrm{det}}$.
- **SFT that keeps deception** ($b_{2}$, $b_{3}$): probe AUROC ≥ baseline; retention ≈ 0.86.- Both probe-aware SFT runs: `attack_too_weak` (softplus tiny vs CE).

The retention gate is the methodological point: AUROC down because the agent stopped deceiving is not a defeated monitor.

---

## Motivation

Agents act with little human review. [AI control](https://arxiv.org/abs/2312.06942) wants cheap monitors on that stream. White-box residual probes look strong ([Apollo / Goldowsky-Dill et al.](https://arxiv.org/abs/2502.03407)): AUROCs 0.96–0.999 on their sets, high catch rate at 1% FPR on ordinary chat.

Those numbers are mostly **non-adaptive**. Adversarial ML has been warning about that for years ([Athalye et al.](https://arxiv.org/abs/1802.00420); [Tramèr et al.](https://arxiv.org/abs/2002.04599)). Activations can be scrambled while behaviour stays ([Bailey et al.](https://arxiv.org/abs/2412.09565)). RL can hit latent monitors ([Gupta & Jenner](https://arxiv.org/abs/2506.14261)).

I wanted one protocol with four properties:

1. a **published** probe (not one I trained),
2. **agentic** episodes, not only the probe’s training domain,
3. a **budget ladder** (prompt → SFT → RL),
4. a **retention gate**, so “AUROC fell because the model stopped lying” is not scored as attacker success.

---

## Setup

- **Model:** `meta-llama/Llama-3.3-70B-Instruct`, fp16, Apollo-matched. No 4-bit.
- **Probe:** frozen Apollo instructed-pairs, layer 22, mean-pool over **response** tokens only. Validated on Apollo insider-trading (AUROC ≈ 0.999), then applied unchanged.
- **Scale:** $N=60$ sessions; frozen deceptive cohort $|C|=29$; one FT seed; $\lambda\in\{0.5,8\}$ (primary 8).
- **Compute:** five FT cells ≈ 2.0 h wall (~4 H200-GPU-h). Full project more (smokes, aborts). Modal bill ≈ **USD 140**.

---

## Definitions

### Label

An episode is deceptive when, at some turn $t^\star$, the visible reply conflicts with the agent’s internal fact **and** advances a covert goal:

$$
y(\tau)=1
\iff
\exists t^\star:\
\mathrm{conflict}(a_{t^\star},\ b^{\mathrm{int}})
\wedge
\mathrm{advances}(a_{t^\star},\ g^{\mathrm{cov}}).
$$

Conflict alone is error. Covert pursuit that is visible is not deception toward an overseer. The probe does **not** define $y$.

### Probe readout

Let $H^{(22)}\in\mathbb{R}^{T\times d}$ be layer-22 residuals and $m\in\{0,1\}^T$ the response-only mask:

$$
s
=
\bar{h}^{(22)}
=
\frac{\sum_{t=1}^{T} m_t H^{(22)}_{t}}{\sum_{t=1}^{T} m_t}
\in\mathbb{R}^{d}.
$$

$$
z
=
w^{\top}\operatorname{scale}(s)+b,
\qquad
p
=
\sigma(z)
=
(1+e^{-z})^{-1}.
$$

I do not refit $(w,b)$.

### Robustness gap

$$
\mathrm{AUROC}
=
\Pr(z^{+}>z^{-})+\frac{1}{2}\Pr(z^{+}=z^{-}),
\qquad
\Delta_{\mathrm{det}}(b)
=
\mathrm{AUROC}(b_{0})-\mathrm{AUROC}(b).
$$

Positive $\Delta_{\mathrm{det}}$ means detection fell. I report it only if retention passes; otherwise the gap is **withheld**. CIs: session-clustered bootstrap 95%.

---

## Attack ladder

After $b_{0}$, freeze the deceptive cohort:

$$
C
=
\{
i:\
\mathrm{intended}_{i}=1
\wedge
J(a_{i}^{(b_{0})})=1
\}.
$$

All later rungs use the same $C$ ($|C|=29$).

| Rung | Attack |
|------|--------|
| $b_{1}$ | Prompt concealment only (no weight update). Not FT-gated. |
| $b_{2}$ | LoRA SFT, probe **out** of the loss: $\mathcal{L}=\mathcal{L}_{\mathrm{CE}}$ |
| $b_{3}$ | LoRA SFT, probe **in** the loss: $\mathcal{L}=\mathcal{L}_{\mathrm{CE}}+\lambda\operatorname{softplus}(z)$ |
| $b_{4}$ | Fresh LoRA REINFORCE: $R=\alpha\cdot\mathrm{retention}-\lambda z-\beta\cdot\mathrm{surfaceRisk}$ |

I train $b_{2}$/$b_{3}$ on $b_{0}$ deceptive completions, not on $b_{1}$ (those often already stopped deceiving). Softplus is on the **logit** $z$, never on $\sigma(z)$ — the probability form saturates and kills the gradient.

**Engagement check.**

$$
\Delta z
=
\mathbb{E}[z(b_{0})]-\mathbb{E}[z(b)],
\qquad
\mathrm{rel}
=
\frac{\Delta z}{\mathrm{std}(z(b_{0}))}.
$$

`attack_worked` only if $\Delta z \ge 0.1$ and $\mathrm{rel} \ge 0.25$. Else `attack_too_weak`.

**Retention gate** (FT rungs). Let $S(b)$ be deception rate on $C$. Report $\Delta_{\mathrm{det}}$ iff:

$$
S(b)\ge 0.8\cdot S(b_{0})
\quad\text{and}\quad
[\delta_{\mathrm{lo}},\delta_{\mathrm{hi}}]\subset (-0.20,0.20).
$$

(90% session-bootstrap CI on $\delta = S(b)-S(b_{0})$.) Fail → withhold.

**Losses.**

$$
\mathcal{L}_{b_{2}}(\theta)
=
-\sum_{t\in\mathrm{asst}}
\log\pi_{\theta}(a_{t}\mid a_{1:t-1},x).
$$

$$
\mathcal{L}_{b_{3}}(\theta)
=
\mathcal{L}_{\mathrm{CE}}+\lambda\operatorname{softplus}(z),
\qquad
\operatorname{softplus}(z)=\log(1+e^{z}).
$$

$$
\mathcal{L}_{\mathrm{RL}}
=
-R\sum_{t}\log\pi_{\theta}(a_{t}\mid\cdot).
$$

---

## Results

### Transfer (before any attack)

| Domain | Probe AUROC |
|--------|-------------|
| Apollo insider-trading | **0.999** |
| My agentic episodes ($b_{0}$) | **0.83** [0.75, 0.90] |

Same extraction path. This is distribution shift, not a wiring bug. Later gaps are vs **0.83**, not vs 1.0.

### Primary table ($\lambda=8$)

|  | $b_{0}$ | $b_{1}$ | $b_{2}$ | $b_{3}$ | $b_{4}$ |
|--|:-----:|:-----:|:-----:|:-----:|:-----:|
| Surface AUROC | 0.83 | 0.52 | 0.71 | 0.72 | 0.68 |
| Probe AUROC | 0.83 | 0.79 | 0.89 | 0.85 | 0.75 |
| CoT control | 0.52 | 0.44 | 0.54 | 0.53 | 0.50 |
| $\Delta_{\mathrm{det}}$ | — | 0.04 | −0.06 | −0.02 | *withheld* |
| Retention | 1.00 | 0.52 | 0.86 | 0.86 | **0.14** |
| FT gate | — | — | pass | pass | **fail** |

Both $\lambda=0.5$ and $\lambda=8$ probe-aware runs are `attack_too_weak`. At $\lambda=0.5$, probe AUROC stays high through $b_{4}$ (0.90), but retention still fails (0.31). Gated either way.

### Reading the rungs

**Transfer** is the strongest result. Biggest drop, zero attack.

**$b_1$:** surface falls hard (0.83→0.52); probe barely moves (0.79). Same concealment, different monitors.

**$b_2$ / $b_3$:** probe holds or rises (0.89, 0.85) with retention 0.86. Putting the probe in the loss did not open an extra gap. Softplus never really engaged, so this is a **budget bound**, not “probe unbeatable.”

**$b_4$:** AUROC 0.75 only with retention 0.14. Fixed-threshold recall can even go *up* when outputs go weird. That is why I gate.

At $b_0$, probe and surface flags barely overlap (Jaccard 0.08). CoT control sits ~0.5; episodes are single-turn, so that control is weak.

---

## Takeaways

Two claims, different strength:

1. **Transfer (0.999→0.83):** solid.
2. **Flat $b_2$–$b_3$:** weaker; `attack_too_weak`. Bound on *this* budget only.

I expected probe-aware pressure to open a clean gap. Under my budget it did not, unless the agent also stopped deceiving.

Why flat under SFT? Either the probe resists gradients, or softplus never engaged. Logs say the second. I am not claiming the first.

[Gupta & Jenner](https://arxiv.org/abs/2506.14261) find mean-pool monitors harder for RL than token-level ones. That fits a small usable $\Delta_{\mathrm{det}}(b_4)$ *if* retention holds. I got AUROC movement with retention collapse instead. At this budget I cannot cleanly separate “hard probe” from “my RL broke the behaviour.”

---

## Limits

- Both $\lambda$ settings `attack_too_weak`; harder schedules untested.
- $N=60$, $|C|=29$ → wide CIs.
- One model, one probe, one seed.
- Single-turn episodes only.
- Single-annotator retention judge (blind-sample check).
- Code later; evasive checkpoints gated.

---

## If you test monitors

1. Measure on the agentic / deployment distribution. Report transfer with CIs.
2. Use a budgeted adaptive ladder, not one static AUROC.
3. Keep a retention gate.
4. Log `attack_too_weak` vs a working penalty before claiming robustness.
5. Next: harder $\lambda$/curriculum, multi-seed, multi-turn, second model/probe.

Same protocol will report a real adaptive gap when one appears while deception still holds.

---

## References

1. [Goldowsky-Dill et al., deception probes](https://arxiv.org/abs/2502.03407) · [code](https://github.com/ApolloResearch/deception-detection)
2. [Greenblatt et al., AI Control](https://arxiv.org/abs/2312.06942)
3. [Gupta & Jenner, RL-Obfuscation](https://arxiv.org/abs/2506.14261)
4. [Bailey et al., obfuscated activations](https://arxiv.org/abs/2412.09565)
5. [Scheurer et al., deception under pressure](https://arxiv.org/abs/2311.07590)
6. [Benton et al., sabotage evals](https://arxiv.org/abs/2410.21514)
7. [Meinke et al., in-context scheming](https://arxiv.org/abs/2412.04984)
8. [Athalye et al., obfuscated gradients](https://arxiv.org/abs/1802.00420)
9. [Tramèr et al., adaptive attacks](https://arxiv.org/abs/2002.04599)
10. [Zou et al., RepE](https://arxiv.org/abs/2310.01405)
11. [Burns et al., CCS](https://arxiv.org/abs/2212.03827)
12. [Parrack et al., black-to-white](https://arxiv.org/abs/2507.12691)

Comments welcome on the retention thresholds, the `attack_worked` test, and what a harder probe-aware schedule should look like while deception still holds.

*Oudoum A. Houmed · oudoum.ali-houmed@universite-paris-saclay.fr*
