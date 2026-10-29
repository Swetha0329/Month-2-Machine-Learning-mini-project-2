# PhishShift Dataset Source Manifest

## Dataset information

- **Dataset name:** Phishing Email Dataset
- **Repository:** zefang-liu/phishing-email-dataset
- **Dataset page:** https://huggingface.co/datasets/zefang-liu/phishing-email-dataset
- **Original source identified by the dataset card:** Phishing Email Detection, shared by Cyber Cop on Kaggle
- **Original source page:** https://www.kaggle.com/datasets/subhajournal/phishingemails
- **License listed on Hugging Face:** GNU Lesser General Public License v3.0 (LGPL-3.0)

## Use in PhishShift

The dataset is used to develop and evaluate a machine-learning model that classifies email messages as legitimate or phishing.

The project prepares the source records into a structured dataset for model training and evaluation. Labels are mapped to the project's legitimate and phishing categories.

## Project dataset files

- `phishshift_dataset.csv`
- `phishshift_prepared.csv`

These are project data files. Their exact relationship to the downloaded source file and the transformations applied should be confirmed from the project's preparation scripts and records.

## Limitations and attribution

The license above is the license listed on the Hugging Face dataset card. The original source's terms and the provenance of the underlying email records should also be checked before redistribution or other uses.

The project should not claim that all messages represent current phishing campaigns or that similarity-based groups are verified campaign identifiers.

## Verification record

- Dataset source identified from the Hugging Face dataset card.
- Local dataset lineage and exact source version: pending verification.
- Download date and source-file checksum: not yet recorded.
## Local preparation evidence

The source retrieval script is `src/download_dataset.py`.

It retrieves records from the Hugging Face dataset API for
`zefang-liu/phishing-email-dataset`, reads the `Email Text` and
`Email Type` fields, maps labels to `legitimate` and `phishing`,
removes duplicate message text, and saves the result as
`data/phishshift_dataset.csv`.

The script derives URL features from message text without opening or
visiting URLs. It sets campaign groups to `unknown_source_campaign`
because genuine campaign identifiers are not provided by the source.

The exact source revision and license terms for the underlying records
still require confirmation.
## Local integrity record

- Verification date: 2026-10-29 13:02:20
- File: `data/phishshift_dataset.csv`
- SHA-256: `A21748281A809CE4C43FC26D470F355AB902D731FF623C6567FF12892892D0C7`
- Note: This records the current local file's checksum; it does not establish the original download date or source revision.
