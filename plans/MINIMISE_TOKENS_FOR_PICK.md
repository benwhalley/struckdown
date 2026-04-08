# Minimise tokens for constrained slots (pick, bool, int, etc.)

## Problem

Constrained slot types (pick, bool, int, number, date) have predictable, short outputs.
Without a max_tokens cap, models may generate unnecessary tokens before producing the
tool call, wasting cost and latency.

## Previous attempt

`compute_optimal_max_tokens()` calculated tight per-slot limits based on action type
and injected them in `segment_processor.py` before each LLM call. This was removed
because:

- Some models (e.g. gpt-5-mini) exceeded the tight limit (50 tokens for bool) before
  generating any response, causing `BadRequestError`.
- The overhead constant (40 tokens for tool calling) was a guess and varied across
  models and providers.
- The saving per call was small relative to prompt tokens, making it a marginal
  optimisation with a high breakage risk.

## If revisiting

- Consider making it opt-in per slot or per pipeline rather than automatic.
- Test against all target models -- different providers have different tool-call overhead.
- Use a generous safety margin (2-3x the theoretical minimum) rather than a tight fit.
- Consider using `logit_bias` or `logprobs` to steer constrained outputs instead of
  token limits, where the provider supports it.
