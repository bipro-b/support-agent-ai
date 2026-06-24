"""SupportSession — a thin orchestrator that ties Phase 1 + Phase 2 together.

This is the first time the pieces become one flow. Each `ask()` call:

    1. RETRIEVE   relevant KB chunks for the question        (Phase 1)
    2. COMPACT    summarize old turns if over the budget     (Phase 2)
    3. ASSEMBLE   persona + long-term memory + recent turns + summary + sources
    4. GENERATE   call Claude (with the system prefix cached)
    5. REMEMBER   extract durable facts -> long-term memory; record the turn

Keep this class in mind: in Phase 3 the inside of `ask()` becomes a LangGraph state
machine (route, call tools, loop), and in Phase 4 it gets wrapped in an API. For now it
is plain procedural code so the data flow is obvious.

It returns a TurnResult so the example can show what happened under the hood — which
chunks were retrieved, whether we summarized, the token/cost/cache numbers. Observability
(Phase 6) generalizes exactly this kind of per-step visibility.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings, get_settings
from ..llm import LLMClient, Usage
from ..rag.retriever import Retriever, build_grounding_context
from .context import ContextAssembler
from .conversation import ConversationMemory
from .store import CustomerMemory, JsonFileMemoryStore, MemoryExtractor, MemoryStore
from .summarizer import Summarizer, compact_if_needed, estimate_tokens


@dataclass
class GenerationResult:
    """What the (overridable) generation step produces for one turn."""

    answer: str
    usage: Usage
    steps: list[str]            # orchestration trace ([] for a plain, toolless turn)
    tools_called: list[str]     # tools actually executed ([] for a plain turn)


@dataclass
class TurnResult:
    answer: str
    retrieved: list[str]        # "source > section" labels of retrieved chunks
    summarized: bool            # did we compact older turns this turn?
    new_facts: list[str]        # facts added to long-term memory this turn
    usage: Usage
    steps: list[str] | None = None       # orchestration trace (Phase 3); None for plain turns
    tools_called: list[str] | None = None  # tools executed this turn (Phase 5 evals use this)
    context: str | None = None           # the retrieved sources the model saw (for groundedness)
    trace: object | None = None          # observability span tree (Phase 6); a tracing.Span
    degraded: bool = False               # Phase 8: a dependency failed; this is a degraded answer


class SupportSession:
    def __init__(
        self,
        customer_id: str,
        *,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        retriever: Retriever | None = None,
        store: MemoryStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or LLMClient(self.settings)
        self.retriever = retriever or Retriever(settings=self.settings)
        self.store: MemoryStore = store or JsonFileMemoryStore(self.settings.memory_dir)

        self.customer: CustomerMemory = self.store.get(customer_id)
        self.conversation = ConversationMemory()
        self.assembler = ContextAssembler()
        self.summarizer = Summarizer(self.llm)
        self.extractor = MemoryExtractor(self.llm)

    def ask(self, question: str, *, extract_memory: bool = True) -> TurnResult:
        # 1. RETRIEVE (Phase 1)
        chunks = self.retriever.retrieve(question)
        retrieved_context = build_grounding_context(chunks)

        # Record the user turn, then compact if the history is now over budget.
        self.conversation.add_user(question)

        # 2. COMPACT (Phase 2)
        summarized = compact_if_needed(
            self.conversation,
            self.summarizer,
            budget_tokens=self.settings.context_budget_tokens,
            keep_recent=self.settings.keep_recent_messages,
        )

        # 3. ASSEMBLE
        assembled = self.assembler.assemble(
            customer=self.customer,
            conversation=self.conversation,
            retrieved_context=retrieved_context,
        )

        # 4. GENERATE — overridable hook. Phase 2 just calls the model; Phase 3's
        #    AgentSession overrides this to run the LangGraph tool loop instead.
        gen = self._generate(assembled)
        answer = gen.answer
        self.conversation.add_assistant(answer)

        # 5. REMEMBER — pull durable facts into long-term memory and persist.
        new_facts: list[str] = []
        if extract_memory:
            for fact in self.extractor.extract(question, answer):
                if self._absorb_fact(fact):
                    new_facts.append(fact)
            if new_facts or self.customer.name:
                self.store.save(self.customer)

        return TurnResult(
            answer=answer,
            retrieved=[f"{c.chunk.source} > {c.chunk.section}" for c in chunks],
            summarized=summarized,
            new_facts=new_facts,
            usage=gen.usage,
            steps=gen.steps,
            tools_called=gen.tools_called,
            context=retrieved_context,
        )

    def _generate(self, assembled) -> GenerationResult:
        """Produce the answer for an assembled context. Override to change HOW.

        Default (Phase 2): a single model call with the stable prefix cached.
        AgentSession (Phase 3) overrides this to run the LangGraph tool loop.
        """
        completion = self.llm.complete(
            assembled.messages, system=assembled.system, cache_system=True
        )
        return GenerationResult(completion.text, completion.usage, [], [])

    def _absorb_fact(self, fact: str) -> bool:
        return self.customer.absorb_fact(fact)

    def history_tokens(self) -> int:
        """Estimated tokens currently held as verbatim history (for the demo)."""
        return estimate_tokens(self.conversation.messages)
