"""Decision-model tour: which struckdown slots can use a decision model, and what comes back.

Runs against Microsoft-Decision-1 on Azure Foundry plus an OpenAI chat model:

    AZURE_OPENAI_ENDPOINT=https://<resource>.cognitiveservices.azure.com \\
    AZURE_OPENAI_API_KEY=... OPENAI_API_KEY=... \\
    uv run python examples/decision_demo.py [deployment-name]

Sections:
  1. slot forms a decision model answers
  2. anatomy of the result object, incl. calibration data
  3. <together>: several questions, one request
  4. mixing a chat model and a decision model in one template
  5. the same slots on a chat model (uncalibrated)
  6. failures: no decision model, fuzzy aliases
"""

import json
import logging
import os
import sys

from struckdown import LLM, LLMCredentials, ModelSpec, complete

sys.stdout.reconfigure(line_buffering=True)  # keep warnings (stderr) in order with output
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
for noisy in ("httpx", "struckdown"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)
logging.getLogger("struckdown.routing").setLevel(logging.WARNING)  # fuzzy-match warnings

DEPLOYMENT = sys.argv[1] if len(sys.argv) > 1 else "Decision1"
DECISION = ModelSpec(
    model_name=f"azure:{DEPLOYMENT}",
    model_type="decision",
    alias="decision",
    display_name="Microsoft-Decision-1",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
)
CHAT = ModelSpec(
    model_name="openai:gpt-4.1-mini", alias="fast", api_key=os.environ["OPENAI_API_KEY"]
)
MODELS = [DECISION, CHAT]
CHAT_LLM = LLM(model_name=CHAT.model_name)
CHAT_CREDS = LLMCredentials(api_key=os.environ["OPENAI_API_KEY"])

COMMENT = (
    "Honestly I've stopped seeing the point of any of this. The deadlines all landed in "
    "the same week, I haven't slept properly in ages, and I don't think anyone would "
    "notice if I just disappeared."
)


def heading(text):
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def run(template, models=MODELS, **ctx):
    return complete(template, {"comment": COMMENT, **ctx}, model=CHAT_LLM,
                    credentials=CHAT_CREDS, registry=models)


def summarise(slot):
    """One line per slot: value, then the probability distribution."""
    d = slot.decision
    if d is None:
        return f"{slot.output!r}  (no decision data: free-text slot)"
    legend = d.get("legend") or {}
    probs = d.get("probabilities") or ({"P(yes)": d["noul"]} if d.get("noul") is not None else {})
    dist = "  ".join(f"{legend.get(k, k)}={v:.2f}" for k, v in probs.items())
    out = getattr(slot.output, "triggered", slot.output)  # halt slots return a verdict
    value = f"{out:.2f}" if isinstance(out, float) else repr(out)
    cal = "calibrated" if d["calibrated"] else "UNcalibrated"
    return f"{value:<14} [{cal}]  {dist}"


def show(res):
    for name, slot in res.results.items():
        print(f"  {name:<12} {summarise(slot)}")
    print(f"  {'total cost':<12} ${res.total_cost:.6f}")


# ---------------------------------------------------------------------------------------
heading("1. Slot forms a decision model answers")
# decide        -> always a decision model; fails if there isn't one
# mode=decision -> opt a pick/bool/score/halt slot into the decision model
# Each slot outside <together> is its own request. The last paragraph before the slot is the
# question; everything earlier (incl. earlier answers) is the state being judged.
res = run("""
Student comment: {{comment}}

Is the student at risk of harm to themselves? [[decide:self_harm]]

What is the comment mainly about? [[decide:topic|teaching,assessment,workload,wellbeing,"cannot tell"]]

Is the student asking for an extension? [[bool:wants_extension|mode=decision]]

Which service should we signpost? [[pick:service|"wellbeing team","personal tutor","module leader",mode=decision]]

How urgently should staff follow up? [[score:urgency|"no follow-up","routine","within a week","same day",mode=decision]]

Is the comment abusive towards staff? [[halt:abusive|mode=decision]]
""")
show(res)


# ---------------------------------------------------------------------------------------
heading("2. Anatomy of the result object")
slot = res.results["urgency"]
print(f"""\
res["urgency"]                     -> {res["urgency"]!r}    (the slot value; same as res.results["urgency"].output)
slot.action                        -> {slot.action!r}
slot.options                       -> {[o.value for o in slot.options]}
slot.decision["type"]              -> {slot.decision["type"]!r}   (noul | choice | score)
slot.decision["calibrated"]        -> {slot.decision["calibrated"]}     (True: from a decision model)
slot.decision["score"]             -> {slot.decision["score"]:.3f}   (probability-weighted level index, 0..n-1)
slot.decision["legend"]            -> {slot.decision["legend"]}
slot.decision["probabilities"]     -> {slot.decision["probabilities"]}
slot.decision["confidence"]        -> {slot.decision["confidence"]}  (0..1, how peaked the distribution is)
slot.decision["model"]             -> {slot.decision["model"]!r}  (as reported by the service)
slot.completion.usage              -> {dict(slot.completion.usage)}
slot.completion._hidden_params     -> {dict(slot.completion._hidden_params)}
slot.completion._cached            -> {slot.completion._cached}""")

print("\nyes/no answers carry P(yes) as 'noul':")
print(f"  res.results['self_harm'].decision -> "
      f"{ {k: v for k, v in res.results['self_harm'].decision.items() if k != 'request'} }")
print("\nchoice answers carry a probability per option:")
print(f"  res.results['topic'].decision['probabilities'] -> "
      f"{res.results['topic'].decision['probabilities']}")
print("\nthe exact request sent is kept for auditing:")
print("  res.results['topic'].decision['request'] ->")
print("   ", json.dumps(res.results["topic"].decision["request"], indent=2)[:600]
      .replace("\n", "\n    "), "...")
print("\nhalt slots give a verdict object (the run stops if triggered):")
print(f"  res['abusive'].triggered -> {res['abusive'].triggered}")
print(f"  res['abusive'].reason    -> {res['abusive'].reason!r}")


# ---------------------------------------------------------------------------------------
heading("3. <together>: independent questions about the same text, one request")
# Inside <together>, decision slots for the same model are sent as ONE request: the text
# before <together> is the state and each slot's own line is its question.
res = run("""
Student comment: {{comment}}

<together>
Is the student at risk of harm to themselves? [[decide:self_harm]]

What is the comment mainly about? [[decide:topic|teaching,assessment,workload,wellbeing]]

How urgently should staff follow up? [[score:urgency|"no follow-up","routine","within a week","same day",mode=decision]]
</together>
""")
show(res)
request = res.results["self_harm"].decision["request"]
print(f"\n  one request, questions: {list(request['questions'])}")
print(f"  state sent: {request['state'][:70]!r}...")
print("  usage/cost is booked on the first slot, zero on the rest:",
      [res.results[k].completion._hidden_params.response_cost for k in res.results])
print("""
  Note: in section 1 each question's state included the earlier questions AND answers
  (see the 'request' above: "...[assistant] True"). That can shift later answers -- compare
  'topic' there and here. Use <together> when the questions should be judged independently.""")


# ---------------------------------------------------------------------------------------
heading("4. Mixing models: decide with the decision model, write with the chat model")
# Slot values are ordinary template variables, so later slots can branch on them.
res = run("""
Student comment: {{comment}}

<together>
Is the student at risk of harm to themselves? [[decide:self_harm]]

How urgently should staff follow up? [[score:urgency|"no follow-up","routine","within a week","same day",mode=decision]]
</together>

{% if self_harm or urgency|default(0) > 2 %}
Write a two-sentence, warm acknowledgement pointing the student to the wellbeing team. [[reply|model=fast]]
{% endif %}
""")
show(res)


# ---------------------------------------------------------------------------------------
heading("5. The same slots on a chat model: same shape, but uncalibrated")
# Without mode=decision, pick/bool/score run on the chat model. .decision still exists so
# downstream code can read either, but all the probability sits on the chosen answer.
res = run("""
Student comment: {{comment}}

Is the student at risk of harm to themselves? [[bool:self_harm]]

What is the comment mainly about? [[pick:topic|teaching,assessment,workload,wellbeing]]

How urgently should staff follow up? [[score:urgency|"no follow-up","routine","within a week","same day"]]
""")
show(res)


# ---------------------------------------------------------------------------------------
heading("6. Failures and warnings")
try:
    run("Comment: {{comment}}\n\nIs this urgent? [[decide:urgent]]", models=[CHAT])
except Exception as e:
    print(f"  decide with no decision model ->\n    {type(e).__name__}: {e}")

print("\n  model=Decisi (partial name) runs, but warns loudly:")
res = run("Comment: {{comment}}\n\nIs this urgent? [[bool:urgent|model=Decisi]]")
print(f"  -> {summarise(res.results['urgent'])}")

try:
    run("Comment: {{comment}}\n\nIs this urgent? [[bool:urgent|model=claude]]")
except Exception as e:
    print(f"\n  unknown model alias ->\n    {type(e).__name__}: {e}")
