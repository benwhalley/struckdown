---
layout: default
title: Decision Models
parent: How-To Guides
nav_order: 3
---

# Decision Models

Decision models -- Microsoft-Decision-1, TypeSafe's Jev -- don't write text. Given some
text and a question with a fixed set of answers, they return a calibrated probability for
each answer, in one fast pass, for a fraction of a cent. They suit triage, routing and
risk scoring: anything you would otherwise do with `pick` or `bool`.

Struckdown sends them `decide` slots, and any `pick`, `bool`, `score` or `halt` slot
marked `mode=decision`. Everything else in the template still goes to your chat model.

## Slot types

| Slot | Decision-model question | Value |
|------|-------------------------|-------|
| `[[decide:x]]` | yes/no | `True` when P(yes) >= 0.5 |
| `[[decide:x\|a,b,c]]` | one of the options | the most likely option |
| `[[score:x\|none,low,high]]` | ordered levels, lowest first | the probability-weighted level index, 0 to n-1 |
| `[[pick:x\|a,b,mode=decision]]` | same as `decide` with options | |
| `[[bool:x\|mode=decision]]` | same as `decide` | |
| `[[halt:x\|mode=decision]]` | yes/no guard | trips when P(yes) >= 0.5 |

`score` also works on a chat model, which answers with a level number. `decide` and
`mode=decision` always need a decision model: the run fails rather than quietly using a
chat model.

The probabilities are on the slot result:

```python
res.results["urgency"].decision
# {'type': 'score', 'score': 2.99, 'calibrated': True, 'confidence': 0.97,
#  'legend': {'0': 'none', ...}, 'probabilities': {'0': 0.0, '1': 0.0, '2': 0.01, '3': 0.99}}
```

Slots answered by a chat model get the same shape with `calibrated: False` and all the
probability on the chosen answer, so downstream code can read either.

## Passing the models

Give `complete()` the models it may use as a list of `ModelSpec`. A decision model is marked
with `model_type="decision"`; struckdown never guesses this from the name.

```python
from struckdown import LLM, LLMCredentials, ModelSpec, complete

models = [
    ModelSpec(model_name="azure:Decision1", model_type="decision", alias="decision",
              api_key=AZURE_KEY, base_url="https://<resource>.cognitiveservices.azure.com"),
    ModelSpec(model_name="openai:gpt-4.1-mini", alias="fast", api_key=OPENAI_KEY),
]
res = complete(template, ctx, model=LLM(model_name="openai:gpt-4.1-mini"),
               credentials=LLMCredentials(api_key=OPENAI_KEY), registry=models)
```

The usual provider prefix picks the endpoint:

| Model name | Endpoint | Auth |
|------------|----------|------|
| `azure:<deployment>` | `<base_url>/providers/microsoft/v1/systemone` (Azure Foundry resource) | `api-key` header |
| `typesafe:jev-latest` | `https://api.typesafe.ai/v1/systemone` | Bearer |
| `<alias>` (no prefix) + `base_url` | `<base_url>/v1/systemone`: a gateway such as a LiteLLM proxy | Bearer |

In Django, set the `AvailableModel` type to "Decision Model" and pick it as the model
set's default decision model.

With exactly one decision model in the list, `decide` slots use it. With several, set
`ModelRegistry.default_decision` or name one on the slot with `model=`.

## Batching with `<together>`

Each decision slot is normally its own request, and it sees the conversation so far,
including earlier answers. Inside `<together>` the slots are independent by definition,
so struckdown sends all the decision slots that go to the same model as **one** request:
the text before `<together>` is the state, and each slot's own line is its question.

{% raw %}
```
Student comment:

{{comment}}

<together>
What is this comment mainly about? [[decide:topic|teaching,assessment,workload,wellbeing,other]]

Is the student at risk of harm to themselves? [[decide:self_harm]]

How urgently should staff follow up? [[score:urgency|"none","routine","this week","today",mode=decision]]
</together>

{% if urgency|default(0) > 1.5 %}
Draft a short acknowledgement to the student. [[reply|model=fast]]
{% endif %}
```
{% endraw %}

Outside `<together>`, struckdown doesn't batch: a later question may depend on an
earlier answer.

## Wording questions

- Outside `<together>`, the **last paragraph** before the slot is sent as the question
  and everything earlier as the state. Put the question in its own paragraph.
- Option names are sent as the descriptions, so make them descriptive, and quote any
  that contain spaces: `"within a week"`.
- Add an abstention option such as `"cannot tell"` where the text may not say.
- Treat scores as thresholds and rankings rather than absolute ratings; phrasing and
  option order move them a little.

See `examples/decision_demo.py` for a runnable version.
