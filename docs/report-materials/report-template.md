================================================================
CAPSTONE PROJECT REPORT — TEMPLATE

> **Status notice (2026-08-24):** This template is a historical drafting scaffold
> only. It contains stale result examples and outdated repo references. Use
> `../validated-results.md` as the current source of primary-study facts and the
> project layout in the repository root as the current file structure.
================================================================

Title Page
  - Project Title: Does PEFT Method Ranking Hold Across Script
    Families? A Data-Efficiency Study on Low-Resource Indic
    Languages
  - Your name, roll number, department
  - Guide/faculty advisor name
  - University name, submission date

----------------------------------------------------------------
S.No.  Chapter                                          Page
----------------------------------------------------------------
1.     Acknowledgement                                    3

2.     Abstract                                           4

3.     List of Figures and Tables                         6

4.     1.  Introduction                                   8
       1.1  Objectives                                    9
       1.2  Background and Literature Survey             10
       1.3  Organization of the Report                   11

5.     2.  System Design and Methodology                 12
       2.1  Proposed Approach                            12
       2.2  Working Methodology                          13
       2.3  Standards / Reproducibility Practices         13
       2.4  System Details                                14
            2.4.1  Software / Tools                       14
            2.4.2  Experimental Pipeline (Data,
                   Model, PEFT Configs)                   25

6.     3.  Cost Analysis (Compute Cost)                   37
       3.1  GPU Time / Resource Usage per Method          37

7.     4.  Results and Discussion                         38

8.     5.  Conclusion & Future Works                      40

9.     6.  Appendix                                       41

10.    7.  References                                     50

================================================================
SECTION-BY-SECTION CONTENT MAP (what goes where)
================================================================

1. ACKNOWLEDGEMENT
   - Thank your faculty advisor, department, anyone who helped
     (standard, personal — write this yourself)

2. ABSTRACT
   - Use the ~155-word abstract already drafted:
     "We investigate whether the relative performance ranking
     of three parameter-efficient fine-tuning (PEFT) methods...
     [full text from earlier draft]"

3. LIST OF FIGURES AND TABLES
   - Figure 1: Learning curves (F1 vs. budget, Hindi + Telugu)
   - Table 1: Summary with 95% CI per method/language/budget
   - Table 2: Ranking table (best method per budget/language)
   - Table 3: Compute efficiency (params, memory, time)
   - Table 4: Collapsed/failed runs summary

4. CHAPTER 1 — INTRODUCTION
   1.1 Objectives
       - State the two research questions:
         (1) Does PEFT ranking change with data budget?
         (2) Does it hold across script families (Hindi/Telugu)?
   1.2 Background and Literature Survey
       - Use the Introduction + Related Work drafts already
         written (LoRA, DoRA, IA3 explained; Frontiers 2025 and
         PROPOR 2026 anchor studies; IndicXNLI/XLM-R background)
   1.3 Organization of the Report
       - One paragraph: "Chapter 2 describes... Chapter 3...
         Chapter 4 presents results... Chapter 5 concludes..."

5. CHAPTER 2 — SYSTEM DESIGN AND METHODOLOGY
   2.1 Proposed Approach
       - The controlled factorial design: 3 methods x 2
         languages x 6 budgets x 3 seeds = 108 runs
   2.2 Working Methodology
       - Data pipeline (notebooks 00-02), hyperparameter search
         (notebook 04), full sweep (notebook 05) — pull from
         methodology-notes.md
   2.3 Standards / Reproducibility Practices
       - Fixed seeds, HF Transformers/PEFT as standard libraries,
         documented LR search, incremental logging
   2.4 System Details
       2.4.1 Software / Tools
           - Python, PyTorch, Transformers, PEFT, Kaggle GPU
             (T4/P100), pandas, scikit-learn
       2.4.2 Experimental Pipeline
           - Base model: XLM-RoBERTa-base
           - Dataset: IndicXNLI (Hindi, Telugu)
           - Budgets: 50, 100, 500, 1000, 2000, 20000
           - PEFT configs: LoRA (r=8), DoRA (r=8), IA3
           - The classifier-head bug and fix (from
             methodology-notes.md) — this is your strongest
             "engineering rigor" evidence, include it in full

6. CHAPTER 3 — COST ANALYSIS
   3.1 List of components and their cost
       - Reframe as: GPU compute time per method (from
         compute_efficiency.csv), e.g.:
         LoRA: 36.1 min total training time, 888K trainable params
         DoRA: 50.2 min, 906K params
         IA3:  44.9 min, 657K params
       - Note: no monetary "component costs" apply here (no
         hardware purchased); substitute compute-hour cost if
         your university requires a dollar figure (Kaggle GPU
         hours are free-tier, note this explicitly)

7. CHAPTER 4 — RESULTS AND DISCUSSION
   - Use the Results section already drafted (headline finding:
     IA3 beats LoRA/DoRA from budget=500 onward, both languages;
     LoRA≈DoRA null result; small-budget instability)
   - Use the Discussion section already drafted (contrast against
     Frontiers 2025 and PROPOR 2026 findings)

8. CHAPTER 5 — CONCLUSION & FUTURE WORKS
   - Conclusion: summarize the answer to both research questions
   - Future Works: full-dataset scale-up, cross-task transfer
     (mentioned earlier, correctly deferred), additional Indic
     languages, larger base encoders

9. CHAPTER 6 — APPENDIX
   - Full results table (experiment_results.csv, all 108 rows)
   - collapsed_runs.csv (documented failures)
   - Full methodology-notes.md content (debugging history)

10. CHAPTER 7 — REFERENCES
    - The verified reference list already compiled (Hu et al. 2021,
      Liu et al. 2022/2024, Conneau et al. 2020, Aggarwal et al.
      2022, Nwaiwu 2025, Nina et al. 2026, Sanh et al. 2019,
      Souza et al. 2020, Dettmers et al. 2023)
