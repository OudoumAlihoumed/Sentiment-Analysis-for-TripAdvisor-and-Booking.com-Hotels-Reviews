A robustness-gap protocol for published white-box probes, with initial empirical bounds on Llama-3.3-70B
Oudoum A. Houmed
Epistemic status. Independent empirical project. I stress-tested a published Apollo deception probe under a budgeted adaptive ladder. The transfer result is solid. The adaptive-attack result is a bound on the budget I tested, not a general robustness theorem: both probe-aware SFT cells were attack_too_weak. Code will be released later; evasive LoRA weights stay gated.
TL;DR
People are proposing deception monitors (especially linear probes on residual-stream activations) as a way to oversee LLM agents when humans cannot read every action. Most published numbers are near-ceiling AUROC on the probe’s own domain, against agents that are not trying to evade the monitor.
I asked a different question:
How much detection survives as adversary budget rises — and when should a drop in detection count as evasion at all?
I ran that question as a protocol on Llama-3.3-70B-Instruct against the published Apollo residual probe (Goldowsky-Dill et al., layer 22, mean-pooled). Main results:
Transfer gap (before any attack). Same extraction path: AUROC 0.999 on Apollo’s insider-trading domain → 0.83 (bootstrap 95% CI [0.75, 0.90]) on my agentic episodes.
Attacks that keep deception do not open a further gap. Probe-agnostic SFT ($b_2$) and probe-aware SFT ($b_3$) both keep retention ≈ 0.86 and probe AUROC at or above the agentic baseline.
RL lowers AUROC only when deception collapses. At $\lambda=8$, probe AUROC falls to 0.75, but retention falls to 0.14. I withhold that gap.
Both probe-aware SFT runs were attack_too_weak. Softplus stayed small next to cross-entropy. So the flat $b_2$–$b_3$ contrast is a bound on this budget, not “the probe is unbeatable.”
I did not build a new probe. I built a measurement: a budgeted adversary ladder + a retention gate so a broken agent is not scored as a defeated monitor.




1. Why this matters
LLMs are increasingly used as agents: multi-step actions, tools, thin human review. In that setting an agent can pursue a covert goal while the transcript looks compliant. That is exactly the setting where AI control proposes cheap monitors that flag suspicious behaviour.
Two monitor families are common:
Black-box: score the transcript (surface classifier, CoT reader).
White-box: score internals. The strongest published examples today are linear probes on residual-stream activations (Goldowsky-Dill et al.; also Zou et al., Burns et al.).
Apollo’s probes report AUROCs in the 0.96–0.999 range on their eval sets, and catch most deceptive responses at a 1% FPR threshold calibrated on ordinary chat. That is impressive. It is also usually measured under a non-adaptive assumption: the agent is not optimising against the probe.
A decade of adversarial ML says that assumption is dangerous (Carlini & Wagner; Athalye et al.; Tramèr et al.). Recent work already shows activations can be obfuscated while behaviour stays (Bailey et al.), and that RL can attack latent monitors (Gupta & Jenner, RL-Obfuscation).
What I wanted — and what I did not find as a ready-made protocol — was:
measure a published white-box probe,
on agentic episodes (not only the probe’s training domain),
under a budgeted adaptive ladder (prompt → SFT → RL),
with a retention gate so “AUROC dropped because the agent stopped lying” is not counted as successful evasion.



3.2 Probe readout (frozen Apollo weights)
Let $H^{(22)} \in \mathbb{R}^{T \times d}$ be layer-22 residual states  and $m \in 0,1^T$ the response-only mask. The probe never sees prompt tokens:
$$s = \bar{h}^{(22)} = \frac{\sum{t=1}^{T} mt H^{(22)}{t}}{\sum{t=1}^{T} mt} \in \mathbb{R}^{d}.$$
With published weights $(w,b)$ and the Apollo scaler:
$$z = D{\mathrm{probe}}(s) = w^{\top}\mathrm{scale}(s) + b, \qquad p = \sigma(z) = (1+e^{-z})^{-1}.$$

2. Setup in one picture
Model. meta-llama/Llama-3.3-70B-Instruct, float16, Apollo-matched loading. No 4-bit.
Hardware. Modal cloud, H200:2 (two NVIDIA H200s).
Probe under test. Published Apollo instructed-pairs residual probe at layer 22, mean-pooled over response tokens only. I do not refiton my data. Extraction is validated first on Apollo insider-trading transcripts (AUROC ≈ 0.999), then applied unchanged to agentic episodes.
Scale. $N = 60$ paired sessions. After baseline, I freeze the deceptive cohort $|\mathcal{C}| = 29$. One fine-tuning seed. $\lambda \in 0.5, 8.0$ (primary $\lambda = 8$).
Compute. The five reported fine-tuning cells (train + re-elicit) took ≈ 2.0 hours wall-clock (~4 H200-GPU-hours). Earlier elicitation / Alpaca calibration / probe extraction were reused. The full project cost more (smokes, aborts, an archived non-primary ladder). Modal billing ≈ $140.





3. Definitions (with the math)

3.1 What counts as deception
An episode $\tau = (x_{1:T}, a_{1:T}, s_{1:T})$ has overseer-visible observations $x_t$, agent utterances $a_t$, and residual states $s_t$. Let $g_{\mathrm{cov}}$ be a covert goal hidden from the overseer, and $b_{\mathrm{int}}$ the scenario-internal fact. With a deterministic scenario judge $J$:
$$ y(\tau) = 1 \iff \exists t^\star:\ \mathrm{conflict}\big(a_{t^\star}, b_{\mathrm{int}}\big) \wedge \mathrm{advances}\big(a_{t^\star}, g_{\mathrm{cov}}\big). $$
Both clauses are required:
conflict alone is just error;
covert pursuit that is transparent is not deception toward an overseer.
Labels are monitor-independent. The probe does not define $y$.
3.2 Probe readout (frozen Apollo weights)
Let $H^{(22)} \in \mathbb{R}^{T \times d}$ be layer-22 residual states ($d = 8192$) and $m \in 0,1^T$ the response-only mask. The probe never sees prompt tokens:
$$ s
\bar h^{(22)}
\frac{\sum_{t=1}^{T} m_t H^{(22)}{t,:}} {\sum{t=1}^{T} m_t} \in \mathbb{R}^{d}. $$
With published weights $(w,b)$ and the Apollo scaler:
$$ z = D_{\mathrm{probe}}(s) = w^\top \mathrm{scale}(s) + b, \qquad p = \sigma(z) = \big(1 + e^{-z}\big)^{-1}. $$
3.3 Primary metric: AUROC
Threshold-free detection is the Mann–Whitney form:
$$ \mathrm{AUROC}
\Pr(z_+ > z_-) + \tfrac{1}{2}\Pr(z_+ = z_-). $$
I still calibrate an operating point for recall diagnostics (probe $\alpha = 0.01$ on Alpaca chat, $n=1000$), but on my agentic cohort deceptive scores sit near that threshold, so recall is jumpy. AUROC is the primary readout. Confidence intervals are session-clustered bootstrap 95% CIs.
3.4 The robustness gap
For detector $D$ and adversary budget $b$:
$$ \Delta_{\mathrm{det},D}(b)
\mathrm{AUROC}_D(b_0)
\mathrm{AUROC}_D(b). $$
Positive $\Delta_{\mathrm{det}}$ means detection fell. The object of study is the curve $b \mapsto \Delta_{\mathrm{det}}(b)$, reported only on rungs that pass the retention gate below. A failing rung has its gap withheld (broken-agent suppression). I never plot a collapsed agent as $\Delta_{\mathrm{det}} = 0$ “success for the defense.”




4. Adversary ladder
I treat the monitor as a defense and the agent as a potential attacker.
Probe-agnostic pressure ($b_2$): suppress surface cues only.
Probe-aware pressure ($b_3$, $b_4$): put $z$ in the training objective.
I call a result “adaptive evasion” only if the probe-aware rung also passes retention.
4.1 Freeze the evaluation cohort
After baseline $b_0$:
$$ \mathcal{C}
\big i :\ \mathrm{intended}_i = 1 \wedge J\big(a_i^{(b_0)}\big) = 1 \big. $$
Every later rung is scored on the same $\mathcal{C}$ ($|\mathcal{C}| = 29$). Membership is written to disk and reloaded; it is not recomputed from fresh judgments.
4.2 $b_1$: Prompt concealment (no weight update)
Append a concealment instruction to the system prompt and re-elicit:
$$ a_i^{(b_1)}
\pi_{\theta_0}\big(\cdot \mid \mathrm{system}_i^{(b_0)}\texttt{conceal}, \mathrm{user}_i\big). $$
No training loss. I report retention for honesty, but $b_1$ is not under the fine-tuning retention gate used for $\Delta_{\mathrm{det}}$ decisions.
4.3 $b_2$: Probe-agnostic SFT
LoRA SFT on baseline deceptive assistant tokens under the concealment framing. Probe absent from the loss:
$$ \mathcal{L}_{b_2}(\theta)
-\sum_{t \in \mathrm{asst}} \log \pi_{\theta}\big(a_t \mid a_{<t}, x\big)
\mathcal{L}_{\mathrm{CE}}. $$
Training uses $a^{(b_0)}$ with $J(a^{(b_0)}) = 1$, not $a^{(b_1)}$ (which often already dropped deception and would teach honesty). After training, re-elicit and score with the same frozen probe.
4.4 $b_3$: Probe-aware SFT
$$ \mathcal{L}_{b_3}(\theta)
\mathcal{L}_{\mathrm{CE}} + \lambda\mathrm{softplus}(z), \qquad \mathrm{softplus}(z) = \log\big(1 + e^{z}\big), $$
with $z$ from the current response mean-pool.
Important detail: I penalise the signed logit, never $\mathrm{softplus}(\sigma(z))$. The probability form saturates and kills the gradient before the penalty can move the probe.
I report $\lambda \in 0.5, 8.0$ (primary $\lambda = 8$). On $\mathcal{C}$:
$$ \Delta z
\mathbb{E}[z_{b_0}] - \mathbb{E}[z_b], \qquad \mathrm{rel}
\frac{\Delta z}{\mathrm{std}(z_{b_0})}. $$
A rung is attack_worked only if $\Delta z \ge 0.1$ and $\mathrm{rel} \ge 0.25$; otherwise attack_too_weak.
Flat AUROC under attack_too_weak is a bound, not a proof that the probe is unbeatable.
Primary contrast: $b_2$ vs $b_3$ (probe absent vs probe in the loss), at matched retention.
4.5 $b_4$: Probe-aware RL
Fresh online LoRA REINFORCE (not continued from $b_3$):
$$ R
\alpha \cdot \mathrm{retention}
\lambda \cdot z
\beta \cdot \mathrm{surfacerisk}, $$
with policy gradient
$$ \mathcal{L}_{\mathrm{RL}}
-R \sum_t \log \pi_{\theta}(a_t \mid \cdot) $$
(typical $\alpha = 1$, $\beta = 0.25$).
Prior expectation from RL-Obfuscation: mean-pooled aggregators resist RL obfuscation relative to token-level monitors, so a small usable $\Delta_{\mathrm{det}}(b_4)$ is what you should expect if retention holds.
4.6 Retention gate (the part I care most about)
Let $S_b$ be deception-success on $\mathcal{C}$ at rung $b$, and $S_0$ the $b_0$ rate. Let $\delta = S_b - S_0$ have a session-clustered bootstrap 90% CI $[\delta_{\mathrm{lo}}, \delta_{\mathrm{hi}}]$.
For fine-tuning rungs $b_2$–$b_4$, $\Delta_{\mathrm{det}}(b)$ is reported iff both:
$$ S_b \ge 0.8 S_0 \qquad\text{and}\qquad [\delta_{\mathrm{lo}}, \delta_{\mathrm{hi}}] \subset (-0.20, 0.20). $$
Fail either condition → capability collapse → withhold the gap.
This is the core methodological claim: AUROC down + agent no longer deceiving ≠ monitor defeated.




5. Results

5.1 Transfer gap before any attack
Same validated extraction path:

Domain
Probe AUROC
Apollo insider-trading
0.999
My agentic episodes at $b_0$
0.83 [0.75, 0.90]

That drop is distribution shift, not a broken pipeline. Every later gap is measured relative to 0.83, not relative to 1.0.
If you were planning to copy Apollo’s near-ceiling number onto agentic deployment logs, this is the result that should stop you.
5.2 Primary cell ($\lambda = 8$)


$b_0$
$b_1$
$b_2$
$b_3$
$b_4$
Surface AUROC
0.83
0.52
0.71
0.72
0.68
Probe AUROC
0.83
0.79
0.89
0.85
0.75
CoT-control AUROC
0.52
0.44
0.54
0.53
0.50
Probe $\Delta_{\mathrm{det}}(b)$
—
0.04
−0.06
−0.02
withheld
Retention (frozen)
1.00
0.52
0.86
0.86
0.14
FT retention gate
—
—
pass
pass
fail

Fine-tuning gates apply at $b_2$–$b_4$. $b_1$ is prompt-only (reported retention, not FT-gated).
Attack-strength note. Both $\lambda = 0.5$ and $\lambda = 8.0$ probe-aware runs are attack_too_weak by the $\Delta z$ / $\mathrm{rel}$ test: softplus stayed small next to CE. So the flat probe AUROC at $b_2$ and $b_3$ only bounds the budget I tested.
5.3 Companion cell ($\lambda = 0.5$)
Probe AUROC stays high through $b_4$ (0.90), but retention at $b_4$ still fails (0.31). So $b_4$ is gated either way. The gate, not AUROC alone, decides what I may report.
5.4 How to read each rung
$b_0$ / transfer. Largest real drop, and it happens before any attack. This is the strongest positive finding.
$b_1$ (prompt concealment). Surface collapses ($0.83 \to 0.52$) because concealment targets the transcript. The probe only dips ($0.79$). Black-box and white-box monitors do not degrade the same way under the same concealment. Retention falls to $0.52$, but this arm is not FT-gated.
$b_2$ / $b_3$ (SFT). Probe AUROC stays at or above the agentic baseline ($0.89$, $0.85$) with retention $0.86$ (pass). Putting the probe in the loss did not open an extra gap beyond surface-only pressure — under a penalty that never fully engaged. Meaningful as a null under this budget. Not meaningful as a universal robustness claim.
$b_4$ (RL). AUROC falls to $0.75$ only when retention collapses to $0.14$. Fixed-threshold recall can even rise at $b_4$ when outputs go degenerate. That is why I gate on retention and lead with AUROC. Withholding $\Delta_{\mathrm{det}}(b_4)$ is the honest call.
Surface vs probe overlap. At $b_0$, probe and surface flags are nearly disjoint (Jaccard 0.08). They are not measuring the same thing.
CoT control. Stays near 0.50–0.54. My episodes are single-turn with no separate reasoning span, so this is a weak CoT baseline, not a full reasoning-trace monitor.




6. What I think this means

Two findings, different strength
Transfer gap ($0.999 \to 0.83$). Strong. Does not depend on attack strength. Appears before any adversary.
Flat $b_2$–$b_3$ under my budget. Weaker. Probe-aware SFT that kept deception did not lower detection further, but both runs were attack_too_weak. Bound on this budget only.
Practically:
Do not copy training-domain AUROC onto agentic traffic.
Do not read a flat SFT contrast as “probes are robust forever.”
Do not celebrate AUROC drops that coincide with retention collapse.

Why detection stayed flat under SFT
A flat $b_2$–$b_3$ line could mean:
(A) the probe resists gradient pressure, or
(B) softplus never really engaged.
Training logs point mainly to (B): the penalty stayed small next to CE. So I am not claiming (A).
This is also why the protocol is useful even when the attack fails: it forces you to log attack_too_weak instead of quietly upgrading a null into a robustness theorem.
Relation to RL-Obfuscation
Gupta & Jenner find token-level monitors highly vulnerable under RL, while holistic aggregators stay stronger. My probe mean-pools response tokens, so I expected a small usable $\Delta_{\mathrm{det}}(b_4)$ if retention held. What I got instead: when RL moves the probe ($\lambda=8$, AUROC 0.75), retention collapses (0.14). That is compatible with “mean-pool is harder to attack while behaviour stays,” and also with “my RL schedule mostly broke the behaviour.” I cannot separate those cleanly at this budget.




7. Limitations (please read these)
Attack strength. Both $\lambda$ settings are attack_too_weak. Stronger probe-aware schedules are untested.
Sample size. $N=60$, $|\mathcal{C}|=29$ → wide CIs (0.83 sits in [0.75, 0.90]).
Scope. One model, one Apollo mean-pooled probe, one fine-tuning seed.
Episode form. Single-turn stacked episodes, not full multi-turn dialogue. CoT control is weak for the same reason.
Judge. Retention judge is a versioned single-annotator heuristic, checked on a blind sample.
Release. Analysis code / configs / synthetic elicitation suite will be released later. Evasive LoRA checkpoints stay gated on purpose.




8. Takeaways for how to test monitors
I started expecting the gap to open sharply once the adversary became probe-aware. What I measured is more constrained: usable adaptive evasion against this mean-pooled probe is limited under my budget — pressure strong enough to move detection tends to break deception first.
If you evaluate deception monitors, I think you should:
Always measure on the deployment / agentic distribution, not only the probe’s training domain — and report the transfer gap with CIs.
Evaluate against a budgeted adaptive ladder, not a single static accuracy number.
Keep a retention gate. Never score a broken agent as a defeated monitor.
Log whether the probe-aware loss actually engaged (attack_too_weak vs a working penalty) before claiming robustness.
Scale next under the same gates: harder $\lambda$ / curriculum, multi-seed, true multi-turn episodes, second model, second probe architecture.
This is a measurement instrument, not an evasion recipe. The same protocol will register a real adaptive gap as soon as one opens while deception still holds.




References
Goldowsky-Dill, Chughtai, Heimersheim, Hobbhahn. Detecting Strategic Deception with Linear Probes. arXiv:2502.03407 · code
Greenblatt, Shlegeris, Denain, et al. AI Control: Improving Safety Despite Intentional Subversion. arXiv:2312.06942
Gupta & Jenner. RL-Obfuscation: Can Language Models Learn to Evade Latent-Space Monitors? arXiv:2506.14261
Bailey, Serrano, Sheshadri, et al. Obfuscated Activations Bypass LLM Latent-Space Defenses. arXiv:2412.09565
Scheurer, Balesni, Hobbhahn. Large Language Models can Strategically Deceive their Users when Put Under Pressure. arXiv:2311.07590
Benton et al. Sabotage Evaluations for Frontier Models. arXiv:2410.21514
Meinke, Schoen, Scheurer, Balesni, Shah, Hobbhahn. Frontier Models are Capable of In-context Scheming. arXiv:2412.04984
Athalye, Carlini, Wagner. Obfuscated Gradients Give a False Sense of Security. ICML 2018. arXiv:1802.00420
Carlini & Wagner. Towards Evaluating the Robustness of Neural Networks. IEEE S&P 2017.
Tramèr, Carlini, Brendel, Madry. On Adaptive Attacks to Adversarial Example Defenses. NeurIPS 2020.
Zou, Phan, Chen, et al. Representation Engineering. arXiv:2310.01405
Burns, Ye, Klein, Steinhardt. Discovering Latent Knowledge in Language Models Without Supervision. ICLR 2023.
Parrack, Attubato, Heimersheim. Benchmarking Deception Probes via Black-to-White Performance Boosts. arXiv:2507.12691
Grattafiori et al. The Llama 3 Herd of Models. arXiv:2407.21783



Comments welcome — especially on:
whether the retention thresholds ($0.8S_0$ and the TOST window) are too strict / too loose,
better tests for attack_worked than $\Delta z$ / $\mathrm{rel}$,
what a stronger probe-aware schedule should look like while still keeping deception.
Oudoum A. Houmed · oudoum.ali-houmed@universite-paris-saclay.fr
