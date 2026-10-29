# PhishShift — Requirements Traceability

## Project
PhishShift — Message Risk Scanner

## Purpose
Classify message text as potentially legitimate or phishing and show risk-related cues. The tool provides decision support, not a guarantee of safety.

## Requirements and Evidence

| ID | Requirement | Evidence | Status |
|---|---|---|---|
| FR-01 | Deduplicate near-identical templates before splitting | data/template_groups.csv; data/template_similarity_candidates.csv; evaluation scripts | Partially verified; inspect preparation order and grouping coverage |
| FR-02 | Compare rules, TF-IDF linear classifier, and a third model | reports/metrics/rules_baseline.json; reports/metrics/fair_model_comparison.json; reports/metrics/model_comparison.json | Evidence available |
| FR-03 | Evaluate generalization to unseen campaigns | reports/metrics/campaign_group_audit.json | Not completed; verified campaign IDs are unavailable |
| FR-04 | Provide risk cues and uncertainty/abstention behavior | Flask dashboard and model implementation | Requires final UI and threshold validation |
| FR-05 | Measure false positives on legitimate urgent messages and robustness drift | reports/acceptance_tests/acceptance_test_evidence.csv | Partial; 14 cases recorded, 2 failures |
| NFR-01 | Do not open, fetch, or execute suspicious content | Application and data-preparation implementation | Verify before final submission |
| NFR-02 | Do not use real credentials, mailbox dumps, or personal information | Dataset and test-data review | Verify before final submission |
| NFR-03 | Clearly communicate that predictions are not guarantees | Dashboard disclaimer and README | Implemented; verify final UI |

## Dataset and Evaluation Limitations
The prepared dataset contains 17,485 messages. Its campaign_group field uses unknown_source_campaign for all records. Similarity-derived groups can help reduce near-duplicate leakage, but they are not verified campaign identities. Therefore, genuine campaign-separated evaluation has not been demonstrated.

The dataset source is documented in data/SOURCE_MANIFEST.md. Exact source revision and licensing/permission for the underlying records still require confirmation.

## Acceptance-Test Evidence
The current evidence file is reports/acceptance_tests/acceptance_test_evidence.csv. It records 14 preliminary synthetic cases: 12 matched expected labels and 2 failed. AT01 is a legitimate password-reset false positive. AT03a is a missed phishing example with a defanged URL. These failures must remain documented until investigated.

## Remaining Work
1. Verify source revision, attribution, and license/permission.
2. Review duplicate removal and splitting order.
3. Obtain verified campaign-labelled test data if available and perform campaign-separated evaluation.
4. Investigate AT01 and AT03a; add robustness and urgent-legitimate-message tests.
5. Validate abstention behavior and safety claims.
6. Add verified references, architecture screenshots, original SRS traceability, and repository details to the final report.
7. Produce and review the final report PDF and deployment artifacts.

## Honest Reporting Rule
Do not describe similarity-group testing as campaign-separated testing. Do not claim unverified safety checks, licensing, or acceptance tests as completed.
