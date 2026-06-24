# Phase 0 — Foundations: How LLM Systems Actually Work

> **Goal of this phase:** stop treating the model as magic. By the end you should be able to
> explain, to a senior interviewer, exactly what happens between "user types a message" and
> "user sees a reply" — and where the cost, the latency, and the failure modes live.
>
> **বাংলা নোট:** আপনি API call করেছেন আগে। এই doc-টা সেই API-র *নিচে* কী ঘটছে সেটা গভীরভাবে
> দেখাবে — কারণ Senior Engineer হিসেবে আপনাকে "কেন" জানতে হবে, শুধু "কীভাবে" না।

Read this, then read `src/support_agent/llm.py`, then run `examples/phase0_hello_claude.py`.

---

## 1. What an LLM actually is

A large language model is, mechanically, a **next-token predictor**. Given a sequence of tokens,
it outputs a probability distribution over what the next token should be. That's it. Everything
else — answering questions, writing code, "reasoning" — is an emergent consequence of doing that
extremely well over a huge amount of text.

Two consequences fall out of this immediately, and they drive almost every production decision:

1. **It generates one token at a time, left to right.** This is why output latency scales with
   output length, why streaming is possible, and why "think step by step" helps (the model can
   condition later tokens on its own earlier reasoning tokens).
2. **It only knows what's in front of it** — its frozen training data, plus whatever you put in
   the prompt *right now*. It has no live access to your database, today's date, or the previous
   API call. **The entire job of an AI engineer is controlling what's "in front of it."**

> **মনে রাখো:** মডেল কিছু "জানে না" আপনার business সম্পর্কে। আপনি prompt-এ যা দেবেন, ততটুকুই
> তার কাছে আছে। RAG, memory, tools — সবই আসলে "মডেলের সামনে সঠিক জিনিস রাখা।"

---

## 2. Tokens — the unit of everything

The model doesn't see characters or words. It sees **tokens**: chunks of text from a fixed
vocabulary. Roughly, in English, 1 token ≈ 4 characters ≈ ¾ of a word. "Customer support" might be
3 tokens; a rare word or non-English text is more tokens per character.

Why you must care about tokens:

- **You are billed per token** — input tokens (your prompt) and output tokens (the reply),
  at different prices.
- **The context window is measured in tokens** (next section).
- **Latency** roughly tracks token count, especially output tokens.

**Never estimate Claude tokens with `tiktoken`** — that's OpenAI's tokenizer and undercounts
Claude by 15–20%+. Use the API's own counter:

```python
llm.count_tokens(messages, system=SYSTEM)   # see llm.py — wraps client.messages.count_tokens
```

This is how you answer, *before* spending money, "will this prompt fit?" and "what will it cost?"

---

## 3. The context window

The **context window** is the maximum number of tokens the model can consider at once — input
**plus** output. For `claude-opus-4-8` it's **1,000,000 tokens**; for `claude-haiku-4-5` it's
200,000.

This is a hard budget, and it's the source of a huge fraction of production AI engineering:

- Your system prompt, the retrieved documents, the conversation history, the user's question, **and**
  the room reserved for the answer all have to fit inside it.
- A long conversation grows every turn. Eventually it threatens the window. **Managing that is
  Phase 2 (context & memory).**
- More tokens in context = more cost and more latency, *even if* they fit. "It fits" is not the
  same as "it's a good idea."

> **বাংলা নোট:** Context window = মডেলের "working memory"-র সীমা। বড় বলেই সব ঢেলে দেবেন না —
> প্রতিটা token-এর দাম আছে, আর বেশি context মানে বেশি latency।

---

## 4. The Messages API and statelessness

You talk to Claude through one endpoint: `POST /v1/messages`. A request is built from three parts:

```python
client.messages.create(
    model="claude-opus-4-8",
    system="You are a support agent for Brecx Store.",   # the standing instructions
    messages=[                                            # the conversation so far
        {"role": "user", "content": "What are your hours?"},
    ],
    max_tokens=1024,                                      # cap on the REPLY length
)
```

The single most important property: **the API is stateless.** The server remembers *nothing*
between calls. There is no session on Anthropic's side. If you want the model to "remember" that
the customer said their name three messages ago, **you resend the whole transcript every single
turn.**

This is not a limitation to work around — it's the model. "Conversation memory," "chat history,"
"the bot remembers me" — all of it is *your* code resending context. Run demo #3 in the example to
see it: two separate calls → the model has amnesia; one call with the history replayed → it
"remembers." Same model, the only difference is what you put in front of it.

This statelessness is also *why the system scales*: any server can handle any request, because all
the state travels in the request. The flip side — you own all the state management — is exactly the
work of Phases 2, 4, and 8.

### The three roles

- **`system`** — standing instructions and persona. Set once per request, highest authority.
  This is where "you are a support agent, answer in 1–2 sentences, never reveal internal policy"
  lives. *Changing the system prompt is changing the program.*
- **`user`** — input from the human (or, later, tool results we feed back).
- **`assistant`** — the model's previous replies, replayed so it can see its own history.

`messages` must start with a `user` turn and generally alternate. When we replay a conversation
(demo #3), we reconstruct the full `user`/`assistant` ladder.

---

## 5. Sampling: how a distribution becomes one token

The model outputs a probability distribution over the next token; **sampling** is how one token
gets chosen. Historically you controlled this with `temperature` (higher = more random/creative,
lower = more deterministic), `top_p`, and `top_k`.

**On Opus 4.8 / 4.7 these sampling parameters are removed** — passing `temperature` returns a 400
error. The modern guidance is: **steer behavior with the prompt, not with sampling knobs.** If you
want determinism, write a tighter prompt and use a lower *effort* (below); if you want variety, ask
for it explicitly. This is a real shift from older models, and a good interview talking point:
*"how do you get reproducible output from Opus 4.8?"* → not `temperature=0` (gone), but prompt
design + structured outputs.

> **মনে রাখো:** পুরোনো অভ্যাস `temperature=0` Opus 4.8-এ কাজ করবে না (400 error)। আচরণ নিয়ন্ত্রণ
> করুন prompt দিয়ে।

---

## 6. Thinking and effort

Modern Claude can **think before answering** — produce internal reasoning tokens that aren't the
final answer but improve it. On Opus 4.x this is **adaptive thinking**: you turn it on, and the
model decides how much to think per request.

```python
llm.complete(messages, thinking=True)   # see llm.py
```

Two related controls:

- **`thinking: {"type": "adaptive"}`** — let the model reason as much as the task needs.
- **`output_config: {"effort": "low"|"medium"|"high"|"xhigh"|"max"}`** — how hard it works overall
  (more thinking, more tool calls, longer output). Higher effort = better on hard tasks, but more
  tokens and latency.

The engineering tradeoff: **thinking costs tokens and time.** A FAQ lookup doesn't need it; a
multi-step "the customer wants a partial refund split across two orders" decision does. We default
thinking **off** in `complete()` and turn it on deliberately. Knowing *when* to spend reasoning is
itself a senior skill — and a recurring theme once we hit cost/latency (Phase 7).

---

## 7. Streaming

Because the model emits tokens one at a time, you can receive them as they're produced instead of
waiting for the whole reply. Two production reasons to almost always stream user-facing replies:

1. **Perceived latency.** A user who sees words appear in 300ms feels a fast product, even if the
   full answer takes 6 seconds. Time-to-first-token is the metric users actually feel.
2. **Timeouts.** Long non-streaming responses can exceed HTTP idle timeouts and silently fail;
   a stream keeps the connection alive. (The SDK will even refuse very large non-streaming requests
   for this reason.)

See `LLMClient.stream()` and demo #2. The cost is complexity — you handle partial output, errors
mid-stream, and you only learn the final token usage at the end.

---

## 8. The cost model

You pay per token, input and output priced separately, per million tokens:

| Model            | Input ($/1M) | Output ($/1M) |
|------------------|-------------:|--------------:|
| `claude-opus-4-8`  | $5.00      | $25.00        |
| `claude-haiku-4-5` | $1.00      | $5.00         |

Three things to internalize:

1. **Output is ~5x input.** Long, rambling answers are the expensive ones. Concise prompts that
   produce concise answers are cheap.
2. **Input includes *everything*** — system prompt, retrieved docs, full history, the question.
   In a long RAG conversation, the input dwarfs the output. This is why **prompt caching**
   (Phase 2) and **context management** matter so much for cost.
3. **Model choice is the biggest lever.** Haiku is 5x cheaper than Opus on input. Routing easy
   requests to Haiku and hard ones to Opus (Phase 7) can cut a bill dramatically — *if* your evals
   (Phase 5) prove quality held. That ordering — measure, then optimize — is deliberate.

`Usage.cost_usd` in `llm.py` computes this for every call so cost is never invisible.

> **বাংলা নোট:** খরচের সবচেয়ে বড় নিয়ন্ত্রক — কোন model, আর কতটা context পাঠাচ্ছেন। Output
> token সবচেয়ে দামি। তাই উত্তর concise রাখা শুধু UX না, cost-ও।

---

## 9. The prompt is the program

In traditional software, behavior is in the code. In an LLM application, a large part of the
behavior is in the **prompt** — especially the system prompt. Demo #5 shows it starkly: identical
model, identical question, but "terse agent" and "warm empathetic agent" produce completely
different replies. You changed the program without touching a line of logic.

This has deep implications you'll feel all the way through the project:

- **Prompts are versioned artifacts**, not throwaway strings. They deserve the same care as code.
- **Prompt changes need evals** (Phase 5) — a tweak that helps one case can silently break ten
  others. "It looks better" is not a regression test.
- **Most "the AI is wrong" bugs are prompt/context bugs**, not model bugs. The first question in
  production is almost always: *what exactly did we put in front of the model?* — which is why
  observability (Phase 6) logs the full assembled prompt.

---

## 10. The anatomy of a production LLM app (the map for everything after)

A single `messages.create` call is the engine. A production system wraps it in layers, and this
project builds them one by one:

```
request
  │
  ▼
[ security / guardrails ]   ← Phase 9: is this input safe? injection? authz?
  │
  ▼
[ retrieval (RAG) ]         ← Phase 1: pull relevant knowledge-base chunks
  │
  ▼
[ context assembly ]        ← Phase 2: history + profile + retrieved docs, within budget
  │
  ▼
[ orchestration ]           ← Phase 3: route, call tools, loop, ask a human (LangGraph)
  │
  ▼
[  the model call  ]        ← Phase 0: THIS phase — the thing in the middle
  │
  ▼
[ reliability wrapper ]     ← Phase 8: retries, timeouts, fallbacks around the call
  │
  ▼
response  ── observed (Phase 6), measured (Phase 5), cost/latency-tuned (Phase 7)
```

Notice the model call is *one box*. Everything that makes the product good, safe, cheap, and
trustworthy is the boxes around it. **That's the whole thesis of this curriculum.**

---

## 11. Interview-angle checklist for this phase

A senior interviewer probing "do you actually understand LLMs?" asks things like:

- *Why is the Messages API stateless, and what does that imply for your architecture?*
  → State lives in your system; any node can serve any request; you own memory management.
- *A conversation gets long and slow and expensive. What's happening and what do you do?*
  → Input tokens grow every turn (full history resent); manage with summarization / caching /
    trimming → Phase 2.
- *How do you get deterministic output from Opus 4.8?*
  → Not `temperature=0` (removed). Prompt design + structured outputs + low effort.
- *When would you enable thinking, and what does it cost?*
  → Hard multi-step reasoning, not lookups. Costs tokens + latency.
- *Where does the cost in an LLM app actually come from?*
  → Mostly input context (history + RAG) and output length; model choice is the big lever.
- *The model gives a wrong answer in prod. First thing you check?*
  → What was actually in the assembled prompt (not "the model is dumb").

If you can answer all of these in your own words, Phase 0 is done.

---

## 12. Exercises (do these before Phase 1)

1. **Run the demo** and read every `[usage]` line. Which call cost the most? Why?
2. In `phase0_hello_claude.py`, switch the model to `claude-haiku-4-5` for demo #1
   (`llm.complete(..., model="claude-haiku-4-5")`). Compare the reply quality and the cost.
3. Make demo #1 stream instead of returning all at once. Notice time-to-first-token.
4. Break statelessness *on purpose*: in demo #3's "with history" block, delete the
   `assistant(...)` line and rerun. Does it still work? Why or why not?
5. Turn on `thinking=True` for a genuinely hard question (e.g. "A customer ordered 3 items, returned
   1, was charged twice for another — what's the refund?") and watch the token count vs. an easy one.
6. **Write it down:** in 5 sentences, explain to an imaginary junior dev why "the chatbot remembers
   our conversation" is a slight lie. If you can do that cleanly, you've got statelessness.

---

**Next:** Phase 1 — RAG & Retrieval Quality. We give the agent a real knowledge base so it stops
inventing store hours, and — more importantly — we learn how to *measure* whether the retrieval is
any good. Tell me when you're ready.
