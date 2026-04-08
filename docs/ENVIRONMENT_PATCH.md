# ENVIRONMENT_PATCH.md — One-shot prescription fix

## What this patch fixes

The original `mldoctor_env/server/environment.py` let an agent call `prescribe`
multiple times until it got a correct one. The episode only ended on a
*correct* prescription, creating a brute-force loophole: an agent could try
every failure mode in sequence and the cost was only −0.10 per wrong guess
until they hit the right one.

## The fix (applied)

The prescribe handler now ends the episode on the **first** `prescribe` action
regardless of correctness. One-shot diagnosis, like a real on-call SRE.

```python
elif name == "prescribe":
    r, _correct = self._do_prescribe(args)
    reward += r
    done = True   # one-shot diagnosis: prescribe always ends the episode
```

## Verification

Enforced by `tests/test_reward_invariants.py`:
- `test_wrong_prescription_ends_episode`
- `test_correct_prescription_ends_episode`
- `test_brute_force_prescribe_expected_value_negative_on_hard`

Run `pytest tests/ -v` — all 38 tests pass.
