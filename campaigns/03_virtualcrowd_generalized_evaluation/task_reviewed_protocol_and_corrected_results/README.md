# Task - Reviewed Protocol And Corrected Results

## Purpose And Boundary

After author parity is complete, let the user review questionable legacy
semantics such as geometric matching under GT-MOT, then define and recompute a
separately named HJ-reviewed protocol. Preserve author-parity outputs unchanged
and record every semantic delta and resulting metric delta.

## Status

- State: draft
- Next action type: attended planning confirmation
- Next authorized action: begin the next session by reviewing the completed T2
  baseline and the candidate protocol decisions with the user. Do not create
  the Layered Design, change code, or run corrected evaluation until the user
  explicitly activates T3 after that confirmation.
- Blocker: user review and explicit activation are not complete. T2 is complete.

## Activation Gate

Activation requires:

1. completed T2 author-parity tables and receipts;
2. a user-reviewed list of protocol semantics to retain, parameterize, or
   replace;
3. a new task-specific Layered Design residence and review before implementation.

Draft candidates are evidence prompts, not accepted changes: direct GT identity
instead of greedy OKS association, presence-aware temporal denominators,
visibility-aware OKS, gap-aware acceleration, and coordinate/name corrections.

## Handoff

The next session starts here and first checks the situation with the user:

1. Reconfirm the fixed baseline: the HJ composition matches the current author
   evaluator; against the historical frozen tables Crowd4D is `108/108` and
   DyCrowd is `107/108`, with only `scene2 / matched ratio` differing because
   current author and HJ both round the raw `0.9856499999999998` mean to
   `0.9856` while the historical table records `0.9857`.
2. Reconfirm the intended product: preserve the author-parity profile and its
   evidence unchanged, then produce a separately named reviewed protocol and
   corrected result set.
3. Ask the user to decide or prioritize the open protocol surface: direct GT
   identity under GT-MOT; whether any retained matching is greedy or globally
   optimal; visibility-aware OKS; presence-aware temporal denominators;
   gap-aware acceleration; and coordinate/metric naming corrections.
4. Confirm T3 owner/boundary, task split and order, completion criteria, and the
   headline-status consumer before requesting explicit activation.

Remain Draft throughout that check. This handoff authorizes discussion and
planning confirmation only; it does not authorize Layered Design creation,
implementation, corrected runs, or any overwrite/rename/reinterpretation of T2
author-parity outputs.
