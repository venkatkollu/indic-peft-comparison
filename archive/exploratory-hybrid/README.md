# Exploratory hybrid work — excluded from the primary study

This folder contains notebooks 07–09 and their results. They explore a hybrid
IA³+LoRA configuration after the completed primary study (`00`–`06`).

These results must not be combined with `results/` or used as evidence in the
primary conclusion. The runs use changing protocols (including batch size,
learning rate, scheduling, and gradient clipping), are split over several CSV
files, and have not been consolidated into one controlled comparison. The
implementation labelled "sequential" also needs verification that both adapter
families are trainable and that its training procedure is genuinely sequential.
