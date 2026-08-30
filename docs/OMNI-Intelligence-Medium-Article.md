# I Gave the Same Problem to Four Frontier AI Systems. Then I Built a System to Make Them Improve One Another.

## OMNI sends one task to multiple frontier models, audits their answers, returns the errors to the group, and repeats the cycle until the result reaches verified convergence—or the economics say stop.

**Brian James Curry | Vector1 Research**

**Research note: This article describes a working, provider-agnostic prototype and a four-response illustrative pilot. The pilot is not a statistically powered leaderboard. The systems are anonymized, only one response time was captured, and the reported scores are human rubric scores supported by deterministic checks. No claim is made that the prototype has established a universal ranking of frontier models.**

---

The first answer scored 95.

The second scored 89.

The third scored 71.

The fourth scored 55.

All four answers addressed the same synthetic business problem. All four recommended a marketing budget totaling one million dollars. All four sounded confident, analytical, and professionally written.

Two of them also manufactured numbers that were not in the evidence.

That was the interesting result.

The difference between the strongest and weakest systems was not fluency. It was epistemic discipline: the ability to recognize what the data did not establish, resist false precision, and still make a useful decision.

It led me to a different way of thinking about frontier AI.

The question may not be which model is best. The more useful question may be how to combine models so that one system's strengths expose another system's errors—and how to keep doing that until the collective result becomes better than any initial answer.

I built a prototype around that idea.

I call it **OMNI: Orchestrated Multi-Model Networked Inference**.

OMNI sends the same task to multiple frontier models simultaneously. Their answers flow into a downstream evaluation layer. That layer checks constraints, scores the answers, identifies unsupported claims, and sends a common feedback packet back to the frontier models. Each model revises its answer independently. The loop continues until the system reaches verified convergence, stops improving, exceeds its cost ceiling, or reaches its maximum number of rounds.

This is not a chatbot council taking a majority vote.

It is a recursive optimization system for model outputs.

---

## The Experiment That Exposed the Problem

I gave four AI systems a small decision problem involving a one-million-dollar marketing budget split across Paid Search, Paid Social, and Email/CRM.

The prompt included reported revenue, customer acquisition cost, one search holdout, one email experiment, limited email capacity, and no controlled evidence for Paid Social. It also included competing executive objectives: the CFO wanted immediate return, while the CMO wanted customer growth without a material revenue decline.

The task required each system to:

1. Produce an allocation totaling exactly one million dollars.
2. Separate reported revenue from incremental impact.
3. Explain what the evidence could and could not establish.
4. Identify three important uncertainties.
5. Recommend an experiment.
6. Stay below 500 words.
7. Avoid inventing missing data.

The problem was deliberately underdetermined. There was enough information to make a defensible provisional decision, but not enough to calculate a mathematical optimum.

That distinction became the test.

The strongest response explicitly refused to convert the search holdout into a marginal return because the prompt did not say exactly how the holdout had been constructed or which revenue base the four-percent decline described. It made a conservative recommendation, stated its uncertainty, and proposed a better experiment.

The weaker responses sounded more quantitative.

One assumed that a 10-percent holdout meant exactly 10 percent of search spending had been removed. It converted that assumption into a $40,000 reduction, applied the four-percent revenue decline to the reported $1.6 million, calculated a $64,000 loss, and declared a 1.6-times marginal ROI.

None of those measurement details had been provided.

Another response repeated that calculation and then applied the email experiment's 12-percent lift to an unspecified revenue base, producing an estimated $130,000 to $150,000 in incremental revenue. It sounded like the most technically sophisticated answer. It received the lowest score.

The polished language made the unsupported inference harder to notice, not less consequential.

That is the central problem OMNI is designed to address.

---

## Why One Frontier Model Is Not Always Enough

Frontier models do not fail identically.

One may be better at causal reasoning. Another may follow formatting constraints more reliably. Another may catch mathematical inconsistencies. A fourth may produce the clearest executive recommendation. Even two models with similar benchmark performance may approach the same ambiguous problem from different directions.

Research on model ensembles supports the broader proposition that multiple language models can outperform an isolated model under the right orchestration scheme.

Mixture-of-Agents uses layered model outputs as inputs to later agents and reported improvements on major language-model evaluation benchmarks [1]. LLM-Blender combines pairwise ranking with generative fusion rather than selecting a single answer [2]. Multi-agent debate and ReConcile use structured interaction to improve reasoning and factual performance [3][4]. FrugalGPT demonstrates that model routing and cascades can improve the relationship between cost and accuracy rather than using the largest model for every request [5].

The production literature is moving in the same direction. Anthropic describes evaluator-optimizer workflows in which one model generates and another evaluates in a feedback loop [6]. OpenAI distinguishes manager-style orchestration from decentralized handoffs and recommends optimizing model selection only after the required accuracy level has been reached [7].

But more models do not automatically create more intelligence.

They can also produce shared hallucinations, defer to the most confident answer, amplify an early framing error, or converge because every model was trained on similar material. Model judges introduce their own problems, including position bias and self-preference [8][9].

The system therefore needs more than plurality.

It needs architecture.

---

## The OMNI Architecture

OMNI divides the problem into eight components.

### 1. The Task Contract

Every run begins with a machine-readable contract containing:

- The complete prompt.
- The evidence models are permitted to use.
- The required output structure.
- Hard constraints.
- A scoring rubric.
- Explicit penalties for unsupported claims.
- Cost, latency, and iteration limits.

The contract matters because a vague objective produces a vague optimization target. If the system is told only to make the answer “better,” the downstream evaluator can reward verbosity, confidence, or stylistic similarity instead of decision quality.

### 2. Simultaneous Frontier Fan-Out

The same task goes to several frontier models at the same time.

The initial answers are independent. Models do not see one another's work during the first round. This protects diversity and prevents the first answer from anchoring the rest of the system.

The prototype includes provider adapters for OpenAI-compatible APIs, Anthropic, and Gemini. Model identifiers, prices, and credentials remain configuration rather than application logic.

### 3. Deterministic Validation

Before another model expresses an opinion, ordinary code checks everything that ordinary code can check.

For the budget problem, that includes:

- Does the allocation total exactly one million dollars?
- Are all three channels present?
- Is the response below 500 words?
- Does it trigger a known unsupported numerical inference?

For a production system, validators could check schemas, totals, date ranges, database constraints, citations, executable code, unit tests, policy rules, or reconciliation against source data.

This separation is important. Language models should not be asked to approximate arithmetic or compliance checks that deterministic software can perform exactly.

### 4. Blinded Cross-Evaluation

The candidates are anonymized and scored by one or more evaluator models.

The judges do not see model identities. Candidate order is randomized for each judge. Scores are bounded by the task rubric, and fatal errors are recorded separately from stylistic criticism.

No single model should be the sole judge of its own work. Multiple evaluators reduce—but do not eliminate—judge bias.

### 5. The Downstream Arbiter

The candidate outputs, deterministic findings, and blinded critiques flow into a downstream arbiter.

The arbiter produces four things:

- Verified common ground.
- Unsupported claims.
- Unresolved disagreements.
- Directives for the next round.

The arbiter is not asked to produce the final answer yet. Its function is to create an information-rich feedback packet that preserves good minority arguments while removing unsupported reasoning.

### 6. Recursive Revision

Each frontier model receives:

- The original task.
- Its own previous response.
- The anonymized responses from the other models.
- The downstream feedback packet.

The models then revise in parallel.

For model i at iteration r, the next answer can be represented as:

**Answer(i, r+1) = Model(i)[Task, Answer(i, r), Peer Answers(r), Arbiter Feedback(r)]**

The process is recursive, but it is not unbounded.

### 7. Verified Convergence

The word convergence can be dangerous here.

If three models repeat the same false claim, they have reached consensus—not truth.

OMNI therefore treats convergence as a compound condition. A result is accepted only when:

- Hard validators pass.
- The best rubric score exceeds a quality threshold.
- Judge disagreement falls below a threshold.
- Frontier outputs converge sufficiently on the decision.
- No unresolved fatal error remains.

Semantic agreement is only one signal. It cannot override failed calculations or unsupported evidence.

### 8. Final Synthesis and Audit

After the loop stops, the downstream system synthesizes the strongest supported elements into one answer. The final answer goes through the validators again.

For consequential applications, a human remains the final decision authority.

---

## What Exactly Is Being Optimized?

A multi-model system should not maximize answer quality without regard to cost.

Let the utility of iteration r be:

**Utility(r) = Quality(r) − λCost(r) − μLatency(r) − γResidualRisk(r)**

Quality comes from the fixed rubric. Cost includes model calls and tokens. Latency is measured directly. Residual risk includes validator failures, judge disagreement, unsupported claims, and unresolved evidence gaps. The coefficients λ, μ, and γ reflect the application.

A high-stakes financial or healthcare workflow should assign a large penalty to residual risk. A low-risk copywriting task may assign more weight to speed and cost.

OMNI stops for one of four reasons:

1. **Verified convergence:** the quality and validation conditions are satisfied.
2. **Quality plateau:** another round produces less improvement than the configured minimum.
3. **Cost ceiling:** expected model expense reaches the run budget.
4. **Maximum rounds:** the system reaches its hard iteration limit.

This converts “keep improving” from an aspiration into an operating policy.

---

## The Economics of Recursive Intelligence

The architecture is more expensive than a single prompt.

With N frontier solvers and J judges, one evaluated round requires approximately:

**N solver calls + N × J judge calls + one downstream arbitration call**

Three solvers and two judges require roughly ten calls per full feedback round. Parallel execution reduces wall-clock time, but it does not eliminate token cost.

That makes routing essential.

OMNI should eventually operate in three modes:

### Fast Mode

One efficient model answers first. The system escalates only if confidence is low or a validator fails.

### Research Mode

Several frontier models run simultaneously through multiple evaluation and revision rounds. This is appropriate for novel analysis, technical design, research synthesis, and strategic decisions.

### High-Stakes Mode

The full mesh runs with source verification, stricter thresholds, domain-specific validators, and mandatory human approval.

The economic goal is not to use every model every time. It is to purchase additional intelligence only when its expected value exceeds its marginal cost.

---

## What the Prototype Does Today

The version accompanying this article is a GitHub-ready Python prototype with no required third-party packages.

It includes:

- Simultaneous asynchronous calls to multiple model providers.
- OpenAI-compatible, Anthropic, and Gemini adapters.
- A replay mode for evaluating saved responses without API access.
- Machine-readable task contracts and scoring rubrics.
- Word-count and allocation-total validators.
- Configurable task-specific risk detectors.
- Blinded and randomized model judging.
- A downstream feedback memo.
- Recursive parallel revisions.
- Quality, disagreement, convergence, cost, plateau, and round stopping rules.
- A final synthesis and validation stage.
- Unit tests and the four-response pilot.

The local replay reproduces the pilot ranking of 95, 89, 71, and 55. It also detects the known unsupported search calculation in the third response while leaving the disciplined first response unflagged.

The current prototype should be understood as an orchestration harness, not proof that recursive model collaboration always improves results. A proper empirical study requires a larger task set, repeated samples, current model versions, complete latency and token accounting, calibrated human judges, and ablation tests.

Those ablations matter.

We need to compare:

- Best single model versus the full mesh.
- Same-model self-consistency versus cross-lab diversity.
- Majority vote versus blinded scoring.
- One-pass synthesis versus recursive feedback.
- Model-only evaluation versus model plus deterministic validation.
- Fixed three-round loops versus adaptive stopping.

Without those comparisons, it would be easy to attribute improvement to the wrong component.

---

## The Larger Idea

Frontier labs are building increasingly capable general models. Most users still consume those models one at a time through isolated interfaces.

That may not be the final architecture.

The models may become cognitive components inside higher-order systems: one proposes, another challenges, deterministic software verifies, a downstream model arbitrates, and the full network iterates against an explicit objective.

The durable asset may not be any individual model.

It may be the orchestration layer that knows which models to ask, how to preserve independent reasoning, how to identify unsupported claims, when to demand another round, and when the marginal value of more intelligence has fallen below its cost.

OMNI is an early prototype of that layer.

The four-system pilot did not show that one frontier lab had solved reasoning and another had not. It showed something more operationally useful: models fail differently, confidence can hide unsupported inference, and an answer can become more persuasive as it becomes less defensible.

A well-designed multi-model system can use those differences as error-correcting structure.

The goal is not consensus.

The goal is verified improvement.

---

## References

[1] Wang, J., Wang, J., Athiwaratkun, B., Zhang, C., and Zou, J. “Mixture-of-Agents Enhances Large Language Model Capabilities.” 2024. https://arxiv.org/abs/2406.04692

[2] Jiang, D., Ren, X., and Lin, B. Y. “LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion.” Proceedings of ACL, 2023. https://aclanthology.org/2023.acl-long.792/

[3] Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., and Mordatch, I. “Improving Factuality and Reasoning in Language Models through Multiagent Debate.” 2023. https://arxiv.org/abs/2305.14325

[4] Chen, J. C.-Y., Saha, S., and Bansal, M. “ReConcile: Round-Table Conference Improves Reasoning via Consensus among Diverse LLMs.” Proceedings of ACL, 2024. https://arxiv.org/abs/2309.13007

[5] Chen, L., Zaharia, M., and Zou, J. “FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance.” 2023. https://arxiv.org/abs/2305.05176

[6] Anthropic. “Building Effective AI Agents.” 2024. https://www.anthropic.com/engineering/building-effective-agents

[7] OpenAI. “A Practical Guide to Building AI Agents.” https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/

[8] “Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge.” 2024. https://arxiv.org/abs/2406.07791

[9] “Self-Preference Bias in LLM-as-a-Judge.” 2024. https://arxiv.org/abs/2410.21819

---

## About the Author

Brian Curry is a Kansas City-based machine learning and AI researcher, data scientist, and founder of Vector1 Research. His work focuses on causal inference, complex-system optimization, agentic orchestration, cognitive AI architecture, and the economics of intelligent systems. He is the creator of Papilon, PyCausalSim, Memory-Node Encapsulation, and a growing body of open-source research on how AI systems reason, remember, coordinate, and change organizations.
