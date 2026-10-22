# Required Data Pipeline

1. Obtain datasets and pretrained assets under their applicable access terms.
2. Decode original inputs in the channel order expected by the analyzer.
   SBI RGB arrays used for landmarks must be converted to BGR before shared
   OpenCV anomaly analysis.
3. Generate raw evidence from reference/forged pairs and available or derived
   masks. Preserve the actual reference matching and failure handling.
4. For SBI, retain only source video paths verified against the active FF++
   training split. Reject validation, test and unmatched sources. This check
   is not an identity-disjointness proof.
5. Preserve region/attribute associations. Assign texture/blur to L3,
   structure to L6, and boundary/color wording to L9. Image-level boundary
   evidence must not be presented as localized regional evidence.
6. Construct compact L12 summaries from supported anomaly categories.
7. Audit matching and HERL candidate token lengths including special tokens.
8. Preserve missing-text validity masks and the original hybrid sampler.

The corrected loader and generator still need to be imported from the server.
This document describes the required sequence, not an implemented public CLI.
