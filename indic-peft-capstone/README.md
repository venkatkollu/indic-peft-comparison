# PEFT comparison capstone

This folder is a self-contained archive of a capstone study comparing LoRA,
DoRA, and IA³ for Hindi and Telugu natural-language inference. It has been
organized so the preserved final artifacts can be reviewed without relying on
the unavailable original experiment directory.

## Start here

- [Project status](docs/PROJECT_STATUS.md) — current scope, findings, and limitations.
- [Results summary](docs/validated-results.md) — the maintained interpretation of the result tables.
- [Results](results/) — preserved primary, summary, and hybrid comparison CSVs.
- [Final report](reports/final_capstone_report.docx) — the generated capstone report.

## Layout

| Location | Contents |
|---|---|
| `docs/` | Current project documentation, references, audit notes, and archived draft material. |
| `results/` | Final CSV exports: 108 primary runs, 36 aggregate rows, and 12 hybrid comparison cells. |
| `figures/` | Four final figures used by the report. |
| `notebooks/` | Preserved experiment-sweep notebook. |
| `scripts/` | Historical generation, validation, audit, and report-building scripts. |
| `reports/` | Final DOCX report outputs. |
| `source/` | Supplied pasted source material. |

## Important reproducibility note

The scripts retain their original absolute paths and expect raw experiment files
that are not included in this archive. Treat the CSV files in `results/` as the
available source data for review. The scripts are preserved as implementation
history; they are not currently portable rerun entry points.

## Reporting policy

Keep the primary validation sweep and hybrid held-out comparison separate. The
summary in `docs/validated-results.md` is the maintained source for claims.
Older prose is retained only in `docs/archive/` for traceability.

`reports/final_capstone_report.docx` is the valid report document. The smaller
`reports/Capstone_Project_Report_final.docx` is a preserved upload note that
describes an earlier 15-page report, not a Word document.
