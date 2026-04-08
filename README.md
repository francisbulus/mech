# 🔬 MECHANISTIC INTERPRETABILITY MASTER PLAN (10 WEEKS, SWE BACKGROUND)

You're not trying to become a theoretical ML researcher. You're trying to become the person who can:

- **understand what's happening inside a neural network** — what features it represents, how information flows, why it produces specific outputs
- **run rigorous interpretability experiments** — form hypotheses, design tests, interpret results, spot illusions
- **build and use the tools** — TransformerLens, SAEs, probes, attribution patching, steering vectors
- **produce public research output** — blog posts or papers that demonstrate real findings

This plan assumes:

- You can code well in Python already
- You have zero or near-zero ML background
- You want first-principles understanding, not hand-waving
- You want to produce real artifacts, not just "complete tutorials"

---

## Philosophy (from Neel Nanda)

> Learn the absolute minimal basics as quickly as possible, then immediately transition to learning by doing research.

Do NOT spend months reading papers before writing code. Mech interp is an empirical science. You learn by running experiments, not by reading about them.

---

## Three stages

| Stage                           | Duration   | Goal                                                           |
| ------------------------------- | ---------- | -------------------------------------------------------------- |
| **Stage 1: Learning the Ropes** | Weeks 1–4  | ML basics, transformer internals, mech interp toolkit          |
| **Stage 2: Mini-Projects**      | Weeks 5–7  | Throwaway 1–5 day research projects, practice fast-loop skills |
| **Stage 3: Full Projects**      | Weeks 8–10 | 1–2 week sprints, deeper skills, public output                 |

---

## Projects you will ship (portfolio anchors)

1. **micrograd + transformer from scratch** — Proves you understand the machinery from first principles
2. **2–3 mini-project write-ups** — Short explorations demonstrating you can run and interpret experiments
3. **1 full research project** — A blog post or short paper extending existing work with real findings

---

## Time commitment

Target: **20–30 hours/week**. The plan produces real artifacts at this pace. More hours means faster progress, but diminishing returns if you're not sleeping on what you've learned.

---

## Tools & setup (do this before Week 1)

- **Python environment**: conda or venv, Python 3.10+
- **GPU access**: Google Colab to start, then rent a cloud GPU (runpod.io or vast.ai) — you'll need one by Week 3
- **LLM subscription**: Gemini 2.5 Pro (free via AI Studio) at minimum. Claude or GPT-5 subscription if budget allows.
- **Code editor**: Cursor ($20/month, worth it) — but do NOT use AI autocomplete during ARENA exercises. Use LLMs only as a tutor when stuck during learning phases.
- **TransformerLens**: `pip install transformer-lens` (install in Week 3)
- **nnsight**: `pip install nnsight` (alternative/complement to TransformerLens, explore in Stage 2)

---

# STAGE 1: LEARNING THE ROPES (Weeks 1–4)

Goal: Get enough foundation that all further learning happens through doing research.

Priority is **breadth over depth**. You want to be functional, not expert. Move on even if you don't feel "done."

---

## WEEK 1 — ML MECHANICS + BACKPROP FROM SCRATCH

### Goal by end of week

You can explain the training loop in plain English — what loss measures, what gradients are, what `.backward()` does, what the optimizer changes — and you've built it yourself.

### Resources

- **3Blue1Brown**: Essence of Linear Algebra (watch all, this is essential)
- **3Blue1Brown**: Neural Networks series
- **Karpathy**: nn-zero-to-hero Lecture 1 (micrograd)
- **StatQuest**: Gradient descent + backprop basics

### Day-by-day

**Day 1 — Linear algebra intuitions (essential foundation)**

- Watch 3Blue1Brown Essence of Linear Algebra chapters 1–7
- Focus on: what vectors really are, linear transformations as matrices, matrix multiplication as composition
- Write a summary in your own words. Feed it to an LLM with an anti-sycophancy prompt and ask for harsh feedback on your understanding
- Output: `notes/week1/day1_linear_algebra.md`

**Day 2 — Linear algebra continued + neural net intuition**

- Finish 3Blue1Brown linear algebra (chapters 8–15, especially eigenvalues, change of basis)
- Watch 3Blue1Brown Neural Networks chapter 1 (what is a neural network?)
- Make sure you understand: SVD and why it works, what changing basis means and why it matters, key differences between low rank and full rank matrices
- Output: `notes/week1/day2_linalg_continued.md`

**Day 3 — Backprop from first principles**

- Watch 3Blue1Brown backprop videos
- Watch StatQuest gradient descent + backprop
- Before coding: write one page answering — What are parameters? What is loss? What is a gradient? What does an optimizer step do?
- Output: `notes/week1/day3_training_story.md`

**Day 4 — Build micrograd (autodiff engine)**

- Follow Karpathy's micrograd lecture
- Build the scalar `Value` class with `.data`, `.grad`, parent tracking, `.backward()` with topological sort
- Implement ops: `+`, `*`, `relu`
- Write finite-difference gradient checks
- Output: `micrograd/value.py`, `micrograd/tests/test_gradcheck.py`

**Day 5 — micrograd MLP + XOR**

- Build a tiny MLP using your Value class
- Train it on XOR to convergence
- Document what broke and why
- Output: `micrograd/mlp.py`, `micrograd/xor_train.py`, `notes/week1/day5_what_broke.md`

**Day 6 — Translate micrograd → PyTorch**

- Write a tiny regression in PyTorch: compute loss, `.backward()`, print `.grad`, optimizer step, confirm loss decreases
- Compare to your micrograd — map every concept
- Output: `pytorch_basics/autograd_sanity.py`

**Day 7 — Basic training loop + GPU awareness**

- Write a minimal MNIST training loop in PyTorch (no frameworks, no abstractions)
- Log loss, throughput (examples/sec), GPU memory
- Benchmark CPU vs GPU matmul at different sizes — understand when GPU actually helps
- Output: `training_bench/train_mnist.py`, `notes/week1/day7_gpu_basics.md`

### Week 1 success criteria

- You can explain `.backward()` without hand-waving
- You built micrograd and made XOR learn
- You have a clean PyTorch training loop and can explain each line
- Linear algebra feels like a language you're learning, not a mystery

---

## WEEK 2 — TRANSFORMERS FROM SCRATCH

### Goal by end of week

You can explain every component of GPT-2 and have coded a working transformer from scratch.

### Resources

- **Neel Nanda**: "What is a Transformer?" video (the one you're watching!)
- **Neel Nanda**: Transformer video tutorials (start-from-basics series)
- **ARENA Chapter 1.1**: Transformer from scratch coding tutorial
- **Reference**: Put "A Mathematical Framework for Transformer Circuits" (Elhage et al.) in an LLM context window and have it generate exercises to test your intuitions

### Day-by-day

**Day 8 — Transformer architecture: the big picture**

- Watch Neel Nanda's transformer walkthrough videos
- Draw out the full architecture: embedding → positional encoding → attention → MLP → layer norm → unembedding
- Make sure you understand: tokens and tokenization, the residual stream, why causal attention exists (you already know this one!)
- Output: `notes/week2/day8_transformer_architecture.md` (include your own diagram)

**Day 9 — Attention from first principles**

- Understand Q, K, V matrices — what they compute and why
- Understand multi-head attention — why multiple heads, what the head dimension is, how heads are independent computations
- Key insight to internalize: attention is a soft lookup. Q asks "what am I looking for?", K says "what do I contain?", V says "what do I send if selected?"
- Use an LLM to quiz your understanding. Summarize back to it, get harsh feedback.
- Output: `notes/week2/day9_attention_deep_dive.md`

**Day 10 — The residual stream and MLP layers**

- Understand the residual stream as the "communication channel" — each layer reads from and writes to it
- Understand MLP layers: what they do (nonlinear transformation), how they differ from attention
- Understand layer norm and why it's there
- Key insight: the residual stream means every component can be analyzed somewhat independently
- Output: `notes/week2/day10_residual_stream_mlp.md`

**Day 11–13 — Code a transformer from scratch (ARENA 1.1)**

- Work through ARENA Chapter 1.1
- Code every component yourself. Do NOT copy-paste from LLMs. Use LLMs only as a tutor when genuinely stuck.
- Components to implement: token embedding, positional embedding, attention (single head → multi-head), MLP, transformer block, full GPT-2 architecture
- Load pretrained GPT-2 weights and verify your implementation produces the same outputs
- Output: `transformer_from_scratch/` (full implementation)

**Day 14 — Consolidation**

- Write up your understanding of the full transformer in your own words
- Feed the write-up + "A Mathematical Framework for Transformer Circuits" to an LLM. Ask it to quiz you on the connections between your understanding and the paper's framework.
- Make a "things I'm still confused about" list — this seeds future exploration
- Output: `notes/week2/day14_transformer_understanding.md`, `notes/week2/confusions.md`

### Week 2 success criteria

- You can explain every component of GPT-2 and why it exists
- You have a working transformer implementation that loads real weights
- You understand the residual stream abstraction (this is crucial for mech interp)

---

## WEEK 3 — MECH INTERP TECHNIQUES + TOOLING

### Goal by end of week

You can use TransformerLens to run basic interpretability experiments and understand the core techniques.

### Resources

- **ARENA Chapter 1.2**: Interpretability Basics (prioritize first 3 sections: tooling, direct observation, patching)
- **Ferrando et al.**: Overview of key mech interp techniques (use as reference, don't read cover to cover — put in LLM context and ask questions)
- **TransformerLens docs**: Core library you'll use for experiments
- **Neel Nanda's research walkthroughs**: YouTube channel — watch at least one to see the exploration mindset in action

### Day-by-day

**Day 15–16 — TransformerLens basics (ARENA 1.2, first 3 sections)**

- Work through ARENA 1.2: tooling, direct observation, and patching sections
- Learn to: load a model in TransformerLens, run it with cache, inspect activations at any layer, look at attention patterns, do basic logit attribution
- Output: `mech_interp_basics/arena_1_2_notes.py` (annotated notebook)

**Day 17 — Activation patching**

- Understand the technique from first principles: run the model on two inputs, swap an activation from one into the other, see what changes
- Why it matters: this is how you causally test which components matter for a behavior
- Implement it yourself on a simple example (e.g., does patching attention head 7.3 change whether the model completes "The Eiffel Tower is in" with "Paris"?)
- Output: `mech_interp_basics/activation_patching.py`

**Day 18 — Linear probes**

- Understand the idea: train a simple linear classifier on a model's internal activations to test if a concept is represented there
- Why it matters: if a linear probe can decode "is this token a city name?" from layer 5's residual stream, the model probably represents that concept linearly at that point
- Implement a simple probe (e.g., probe for part-of-speech or sentiment at different layers)
- Output: `mech_interp_basics/linear_probes.py`

**Day 19 — Logit lens / Direct Logit Attribution**

- Understand logit lens: project intermediate residual stream states through the unembedding matrix to see what the model "would predict" at each layer
- This gives you a layer-by-layer view of how the prediction forms
- Implement it on a few interesting prompts
- Output: `mech_interp_basics/logit_lens.py`

**Day 20 — Steering vectors**

- Understand the idea: find a direction in activation space that corresponds to a concept (e.g., "happiness"), then add or subtract it during inference
- Exercise from Neel's guide: use an LLM API to generate 32 happy prompts and 32 sad prompts, compute the mean activation difference at a middle layer of GPT-2 Small, then steer the model and measure the effect
- Output: `mech_interp_basics/steering_vectors.py`

**Day 21 — SAE basics (ARENA 1.3.2, skim section 1)**

- Understand Sparse Autoencoders (SAEs) conceptually: they decompose a model's activations into interpretable features
- Why they matter: neurons in a model are often "polysemantic" (represent multiple concepts). SAEs find directions that are more monosemantic.
- Learn to use a pre-trained SAE (e.g., from Neuronpedia). You do NOT need to train one.
- Explore some features on Neuronpedia — look at what activates them, see if they make sense
- Output: `mech_interp_basics/sae_exploration.py`, `notes/week3/sae_understanding.md`

### Week 3 success criteria

- You can use TransformerLens fluently to inspect model internals
- You understand and can implement: activation patching, linear probes, logit lens, steering vectors
- You have a working understanding of SAEs and can use pre-trained ones
- You've watched at least one Neel Nanda research walkthrough

---

## WEEK 4 — LITERATURE LANDSCAPE + EXPLORATION PRACTICE

### Goal by end of week

You have a sense of the field, you've read one paper deeply, and you've done your first unstructured exploration of a model.

### Resources

- **Neel Nanda's favorite papers list** (with summaries and opinions)
- **Open Problems in Mechanistic Interpretability** (community literature review — skim, don't read in full)
- **Neuronpedia**: Attribution graphs for Gemma 2B
- **ARENA 1.4.1**: Causal Interventions & Activation Patching (recommended)

### Day-by-day

**Day 22 — Survey the field (breadth, not depth)**

- Skim Neel's favorite papers list. For each, read the abstract and Neel's summary. Don't read full papers yet.
- Put "Open Problems in Mechanistic Interpretability" in an LLM context. Ask it to summarize the major open questions and which ones are most promising.
- Goal: build a mental map of what's been done, what's exciting, what's played out
- Output: `notes/week4/day22_field_landscape.md`

**Day 23 — Deep dive: read one paper carefully**

- Pick one paper that interests you. Good options:
  - "Refusal in Language Models Is Mediated by a Single Direction" (Arditi et al.)
  - "The Geometry of Truth" (Marks & Tegmark)
  - "Scaling Monosemanticity" (Anthropic)
- Read it properly: write a summary, understand the motivation, the method, the results, the limitations
- Have an LLM chat open with the paper in context — ask questions every time you're confused
- Output: `notes/week4/day23_deep_dive_[paper_name].md`

**Day 24 — What's new and what to avoid (Neel's "fads" guidance)**

- Review Neel's advice on fads to avoid:
  - Toy model interpretability on algorithmic tasks — played out
  - Simple circuit analysis via causal interventions on model components — the bar for novelty is now very high
  - Incremental SAE architecture improvements — diminishing returns
- Review what IS exciting:
  - Attribution graph-based circuit analysis
  - Downstream tasks / auditing games
  - Model organisms
  - Reasoning model interpretability
  - Real-world uses of interpretability (e.g., linear probes for monitoring)
- Output: `notes/week4/day24_field_direction.md`

**Day 25–26 — ARENA 1.4.1: Activation patching deep dive**

- Work through ARENA 1.4.1 (Causal Interventions & Activation Patching)
- This deepens your most important technique
- Output: annotated notebook

**Day 27–28 — First unstructured exploration**

- Pick a prompt where a model does something interesting (e.g., a factual recall, a reasoning step, a refusal)
- Spend two days just exploring with the tools you have: look at attention patterns, try logit lens, try patching, look at SAE features
- No hypothesis required. Your goal is to practice the exploration mindset: maximize information gained per unit time. Make a new plot every few minutes.
- If you learn nothing in 2 hours, pivot to a different prompt or technique
- Keep a research log as you go
- Output: `explorations/first_exploration/`, `explorations/first_exploration/research_log.md`

### Week 4 success criteria

- You have a mental map of the mech interp field
- You've read one paper deeply and can explain its contribution
- You know what directions are promising vs. played out
- You've done your first open-ended exploration and kept a research log
- **You are ready for Stage 2. Move on even if you don't feel "done."**

---

# STAGE 2: MINI-PROJECTS (Weeks 5–7)

Goal: Do a series of throwaway 1–5 day research projects. You are practicing the craft of research, not trying to produce groundbreaking results.

Focus on **exploration** and **understanding**. Don't stress about having the best idea or writing things up perfectly.

After each mini-project, spend at least 1 hour on a **post-mortem**: What did you do? What worked? What didn't? What would you do differently? Write it down.

---

## WEEK 5 — MINI-PROJECT 1: REPLICATE AND EXTEND

### Suggested project

**Replicate a key result from "Refusal in Language Models Is Mediated by a Single Direction," then extend it.**

- Replicate: find the refusal direction in a model, show that ablating it removes refusal
- Extend (pick one): try it on a different model, try it on a different behavior (not refusal), test how robust the direction is across different types of harmful prompts
- This is an understanding-heavy project: you have a clear hypothesis to test

### Day-by-day

**Days 29–30 — Setup + replication**

- Read the paper carefully (if you didn't in Week 4)
- Set up the experimental code. Replicate the core finding.

**Days 31–33 — Extension**

- Pick your extension direction
- Run experiments. Keep a research log. Make lots of plots.
- Practice the understanding mindset: before testing your hypothesis, spend 5 minutes brainstorming "what are the ways this could be false?"

**Days 34–35 — Post-mortem + write-up**

- 1 hour post-mortem: what went well, what went poorly, what would you do differently
- Write a short (1–2 page) summary of what you found. This is practice, not publication.
- Output: `mini_projects/01_refusal_direction/`, including research log, code, and summary

---

## WEEK 6 — MINI-PROJECT 2: EXPLORATION-HEAVY

### Suggested project (pick one)

**Option A: Attribution graphs on Neuronpedia**

- Use Neuronpedia's attribution graphs to explore Gemma 2B
- Form a hypothesis about how the model handles some specific behavior
- Test it using other methods (prompting, patching, etc.)
- This practices the exploration mindset — you don't start with a hypothesis

**Option B: Play with taboo/emergent misalignment models**

- Use Bartosz Cywiński's taboo models (models with a programmed secret word)
- Try as many methods as you can to find the secret: logit lens, SAEs, probing, black-box prompting
- What works? What doesn't? Why?

**Option C: Investigate unfaithful chain-of-thought**

- Pick some prompts from "Chain-of-Thought Reasoning In The Wild Is Not Always Faithful"
- Use whatever tools seem appropriate to understand what's actually happening inside the model
- Open-ended exploration

### Day-by-day

**Days 36–38 — Explore**

- No plan required. Maximize information gain per unit time.
- Make a new plot every few minutes. If you learn nothing in 2 hours, pivot.
- Keep a research log.

**Days 39–40 — Hypothesize + test**

- If hunches formed during exploration, try to test them
- Practice skepticism: before testing, brainstorm ways the hypothesis could be false

**Day 41–42 — Post-mortem + notes**

- 1 hour post-mortem
- Short write-up of findings
- Output: `mini_projects/02_[project_name]/`

---

## WEEK 7 — MINI-PROJECT 3: YOUR CHOICE + SKILL GAPS

### Choose your own project

By now you should have some sense of what interests you. Pick something. If you can't decide, here are options:

- **Truth probes**: Replicate "Geometry of Truth" probes on a modern model. How well do they generalize? Can you break them?
- **Steering vector deep dive**: Pick a concept, build a steering vector, systematically test its limits
- **Reasoning model interpretability**: Apply methods from Bogdan et al. "Thought Anchors" to new types of prompts

### Also this week: fill skill gaps

- Review your post-mortems from Weeks 5–6. What skills are weakest?
- Spend 2–3 hours targeted practice on your biggest gap (experiment design? interpreting results? knowing when to pivot?)
- If you haven't yet: set up nnsight as an alternative to TransformerLens. Try running the same experiment in both.

### Output

- `mini_projects/03_[project_name]/`
- `notes/week7/skill_gaps_review.md`

### Stage 2 success criteria

- You've completed 3 mini-projects with research logs and post-mortems
- You can run experiments, interpret results, and get unstuck
- You have a sense of which techniques are useful when
- You know what kinds of problems interest you most
- You have honest self-assessment of your skill gaps

---

# STAGE 3: FULL PROJECTS (Weeks 8–10)

Goal: Work in 1–2 week sprints. Be more ambitious. Practice ideation and distillation. Produce something public.

The default after each sprint is to **pivot** unless the project is going great. Getting bogged down in a failing project is the main risk at this stage.

---

## WEEK 8 — IDEATION + SPRINT 1

### Days 50–51 — Generate and evaluate ideas

- Block 2+ hours. Open a blank doc. Write down at least 20 research ideas. Quantity over quality.
- Draw from: your mini-project post-mortems, confusions from papers, things from Neel's "what's new" list that excited you, idle curiosities you noted
- Rate each idea yourself out of 10
- For your top 3–5, answer:
  - What would success look like?
  - How surprised would I be if nothing interesting happened after a week?
  - What skills/resources does this need?
  - Has this been done before? (Use an LLM with web search to check)
- Pick one and commit to a sprint
- Output: `notes/week8/idea_generation.md`

### Days 52–56 — Sprint 1

- Work on your chosen project for ~1 week
- Keep a detailed research log
- Practice deeper skills:
  - **Prioritization**: before each session, spend 5 minutes deciding the highest-value thing to do
  - **Skepticism**: actively try to falsify your findings
  - **Literature awareness**: use LLM web search to check if related work exists as questions come up
- If it's going nowhere after 3–4 days, it's fine to pivot

### Day 56 — Sprint post-mortem

- What did you learn? What progress on different skills?
- Continue or pivot?
- Output: `sprints/sprint_1/`, `sprints/sprint_1/postmortem.md`

---

## WEEK 9 — SPRINT 2 (OR CONTINUATION) + WRITING PRACTICE

### Days 57–61 — Sprint 2

- Either continue Sprint 1 (if it's going well) or start a new project
- This week, add focus on **distillation**: as you work, think about how you'd explain your findings to someone else

### Days 62–63 — Write-up practice

- Take your best sprint result and write it up as a short blog post draft
- Follow Neel's iterative writing process:
  1. Distill to 1–3 key claims (your contribution)
  2. Write a TL;DR / abstract
  3. Write a bullet-point outline
  4. Make figures (figures are incredibly important!)
  5. Flesh out into prose
- Remember: the reader has far less context than you. Spell everything out.
- Get feedback: share with a collaborator, post in the Mech Interp Discord, or at minimum feed it to an LLM with an anti-sycophancy prompt
- Output: `writing/blog_post_draft_v1.md`

---

## WEEK 10 — SPRINT 3 + PUBLIC OUTPUT

### Days 64–67 — Final sprint

- Either continue previous work or start fresh
- By now, one of your projects should have enough substance for a public post

### Days 68–70 — Polish and publish

- Revise your write-up based on feedback
- Ensure you have:
  - Clear motivation (why should anyone care?)
  - Honest limitations (don't oversell)
  - Randomly selected examples (not just cherry-picked best cases)
  - Reproducible code (shared repo with README)
- Publish on LessWrong, a personal blog, or both
- Share in the Open Source Mech Interp Slack and Mech Interp Discord
- Output: published blog post + clean code repo

---

# POST-PLAN: WHAT NEXT

## If you want to keep going with research

- Continue the sprint cycle: 1–2 week sprints, post-mortems, pivot or continue
- Start generating your own research ideas regularly
- Seek a mentor: apply to MATS (Neel's next cohort applications), SPAR, or cold-email researchers (follow Neel's cold email advice — target first authors, not famous last authors; start small; show proof of work)
- Aim for a workshop paper or Arxiv submission

## If you want to pivot toward research engineering

- Your mech interp knowledge + your infra plan skills = the hybrid research engineer role
- Anthropic, OpenAI, DeepMind all hire for this: building instrumented inference, activation extraction pipelines, research tooling
- Your public mech interp output shows you understand the science; your infra portfolio shows you can build systems

## Key communities

- **Open Source Mechanistic Interpretability Slack**
- **Eleuther Discord** (#interpretability-general)
- **Mech Interp Discord**
- **LessWrong / AlignmentForum** (for posting and reading)

## Key resources to keep returning to

- Neel Nanda's YouTube channel (research walkthroughs, paper walkthroughs)
- Neuronpedia (attribution graphs, SAE feature exploration)
- ARENA tutorials (for specific techniques as needed)
- TransformerLens / nnsight docs

---

# APPENDIX: RESEARCH MINDSET CHEATSHEET

**Exploration phase:**

- Your north star: maximize information gained per unit time
- You don't need a plan. It's okay to be confused.
- Make a new plot every few minutes
- If you've learned nothing in 2 hours, pivot approach. If 2–3 approaches are dead ends, pivot problem.
- Most of your probability mass should be on "something I haven't thought of yet"

**Understanding phase:**

- Your north star: convince yourself a hypothesis is true or false
- The key mindset is skepticism. Every result is false until proven otherwise.
- The more exciting a result, the more likely it's false.
- Before testing: 5-minute timer, brainstorm "ways this could be false"
- Imagine an obnoxious skeptic. What would shut them up?

**Distillation phase:**

- The reader has far less context than you think
- Distill to 1–3 key claims. Everything else serves those claims.
- Figures are incredibly important
- Include randomly selected examples, not just your best cherry-picks
- Acknowledge limitations honestly — competent researchers see through overselling

**General habits:**

- Keep a running research log
- Keep a "highlights" doc for cool findings
- Keep a long-running "idle curiosities" doc — seed future projects
- Post-mortem after every project
- Use LLMs aggressively (with anti-sycophancy prompts for feedback)
- Audit your time occasionally — where did it go? What was inefficient?
