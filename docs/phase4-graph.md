# The investigation flow as a LangGraph graph

Generated from `app/agent/graph.py` (`build_graph(...).get_graph().draw_mermaid()`), not drawn by hand. Dashed lines are the conditional edges: no documents skips `investigate`; only an escalated claim reaches `escalate`.

## The flow (what the tests compare with the hand-written version)

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	extract(extract)
	review(review)
	investigate(investigate)
	finalize(finalize)
	escalate(escalate)
	__end__([<p>__end__</p>]):::last
	__start__ --> extract;
	extract --> review;
	finalize -.-> __end__;
	finalize -.-> escalate;
	investigate --> finalize;
	review -.-> finalize;
	review -.-> investigate;
	escalate --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

## With the pause for a person (`human_review=True`)

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	extract(extract)
	review(review)
	investigate(investigate)
	finalize(finalize)
	escalate(escalate)
	human_review(human_review)
	__end__([<p>__end__</p>]):::last
	__start__ --> extract;
	escalate --> human_review;
	extract --> review;
	finalize -.-> __end__;
	finalize -.-> escalate;
	investigate --> finalize;
	review -.-> finalize;
	review -.-> investigate;
	human_review --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```
