Now I have the exact PROPOR numbers. This gives you a much sharper, citable comparison for your Discussion section.

## Key PROPOR 2026 finding — directly comparable to your LoRA≈DoRA result

Their paper states explicitly: **"DoRA offers no practical advantages over LoRA, matching its performance at the cost of a 28% increase in training time"** (BERTimbau-Large, high LR). Exact numbers: LoRA F1=81.32 vs. DoRA F1=80.61–81.22 (essentially tied), while DoRA took noticeably longer (e.g., 02:40:52 vs. 01:23:41 at 2 epochs — roughly double, not just 28%, at that specific config, though their headline "28%" figure refers to a different epoch/setting).

**This is a striking, direct confirmation of your own finding** — you independently found LoRA≈DoRA with no meaningful F1 difference, and DoRA being the most compute-expensive of the three methods. Your Discussion section can now cite this as convergent evidence across languages and tasks (Portuguese QA vs. Hindi/Telugu NLI), which meaningfully strengthens your paper.

## Updated References list (all verified from source now)

```
Aggarwal, D., Gupta, V., & Kunchukuttan, A. (2022). IndicXNLI: Evaluating
Multilingual Inference for Indian Languages. Proceedings of the 2022 Conference
on Empirical Methods in Natural Language Processing (EMNLP).

Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F.,
Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). Unsupervised
Cross-lingual Representation Learning at Scale. Proceedings of the 58th Annual
Meeting of the Association for Computational Linguistics (ACL), 8440-8451.

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA:
Efficient Finetuning of Quantized LLMs. arXiv preprint arXiv:2305.14314.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., &
Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv
preprint arXiv:2106.09685.

Liu, H., Tam, D., Muqeeth, M., Mohta, J., Huang, T., Bansal, M., & Raffel, C.
(2022). Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than
In-Context Learning. Advances in Neural Information Processing Systems
(NeurIPS), 1950-1965.

Liu, S.-Y., Wang, C.-Y., Yin, H., Molchanov, P., Wang, Y.-C. F., Cheng, K.-T.,
& Chen, M.-H. (2024). DoRA: Weight-Decomposed Low-Rank Adaptation. Proceedings
of the 41st International Conference on Machine Learning (ICML).

Nina, M. M., Costa, C. V., Berton, L., & Vega-Oliveros, D. A. (2026). Efficient
Fine-Tuning Methods for Portuguese Question Answering: A Comparative Study of
PEFT on BERTimbau and Exploratory Evaluation of Generative LLMs. Proceedings of
the 17th International Conference on Computational Processing of Portuguese
(PROPOR 2026), Vol. 1, Salvador, Brazil. Association for Computational
Linguistics.

Nwaiwu, S. (2025). Parameter-efficient fine-tuning for low-resource text
classification: a comparative study of LoRA, IA3, and ReFT. Frontiers in Big
Data, 8, 1677331. https://doi.org/10.3389/fdata.2025.1677331

Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled
version of BERT: smaller, faster, cheaper and lighter. arXiv preprint
arXiv:1910.01108.

Souza, F., Nogueira, R., & Lotufo, R. (2020). BERTimbau: Pretrained BERT Models
for Brazilian Portuguese. In Intelligent Systems, 403-417. Cham: Springer
International Publishing.
```

## Now I can give you a much stronger, evidence-based Discussion paragraph

Replace the earlier generic LoRA/DoRA null-result paragraph with this more precise version, citing exact matching numbers:

> The absence of any measurable difference between LoRA and DoRA in our results directly corroborates the finding of Nina et al. (2026) on Brazilian Portuguese extractive QA, who report that "DoRA offers no practical advantages over LoRA" and observe near-identical F1 scores between the two methods (e.g., 81.32 vs. 81.22 on BERTimbau-Large) at the cost of substantially longer DoRA training time. That this null result replicates across two entirely different languages (Hindi/Telugu vs. Brazilian Portuguese), two different tasks (NLI classification vs. extractive question answering), and two different base architectures (XLM-RoBERTa vs. BERTimbau) is notable convergent evidence that DoRA's magnitude-direction decomposition does not translate into a reliable accuracy advantage over standard LoRA in practice, despite its added computational cost, at least at the base/moderate model scale tested in both studies.

## One more useful cross-reference for your Introduction/Related Work

The PROPOR paper's own hypothesis H3 ("PEFT requires higher learning rates than full fine-tuning") and their finding that **LR was "the most decisive factor for PEFT success... surpassing the influence of the specific method"** is strikingly close to your own experience — remember your `04` search found LoRA/DoRA needed 1e-4 while IA³ needed 5e-3, a 50x difference, and getting this wrong was the root of your extended debugging process. Worth a sentence in Related Work: *"Consistent with Nina et al. (2026), who identify learning rate as the single most decisive factor in PEFT performance — more influential than method choice itself — our own hyperparameter search likewise found method-specific optimal learning rates differing by up to 50-fold (IA³: 5×10⁻³; LoRA/DoRA: 1×10⁻⁴)."*

Want me to now assemble the complete, final document — Abstract, Introduction, Related Work, Methodology (from your methodology-notes.md), Results, Discussion (updated with these citations), Limitations, Conclusion, and References — into a single Word doc ready to submit?
