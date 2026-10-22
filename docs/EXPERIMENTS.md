# Experiment Inventory

All corrected comparisons must share data versions, training budget and
checkpoint selection unless their specific variable is intentionally changed.

- Completed: corrected full B/16 model and frame/video evaluation.
- Separate run: four-layer mean aggregation followed by LayerNorm versus FTCA.
- Queued scope: without PFSA; without HERL/PFSA; PH-only; PH+HN; random HN;
  PH+HN+PFSA; Spec-only; Mono+Spec; Reverse/Random/Shared mapping;
  fixed equal and fixed deep-biased level weights.
- Parameter sensitivity, multiple seeds, visualization refresh and corrected
  L/14 reproduction are not automatically supplied by the 13-job queue.

The queue has not been verified as completed from local evidence.
Do not copy historical ablation results into the corrected-results table.

Mean aggregation preserves four-level losses and uses the aggregate for the
L12 projected feature. It is a whole aggregation-block replacement, not merely
uniform attention weights. Attention parameters are retained but unused in the
isolated experiment to preserve initialization order.

Mapping controls change matching-text and HERL candidate assignments together.
PFSA retains original text references and constraints. This limitation must
remain visible in interpreting the mapping comparison.
