# ForgeFold
Fast and Accurate Single Sequence All-Atom Protein Structure Prediction

Project 99: ForgeFold: Fast and Accurate Single Sequence All-Atom Protein Structure Prediction



Team Members:

Casey Mogilevsky (csm70@seas.upenn.edu)

Yuxin Li (grit1021@sas.upenn.edu)

Steven Su (hanqisu@sas.upenn.edu)

Willard Ford (Willard.Ford@pennmedicine.upenn.edu)

Abstract:

This project aims to modernize single-sequence protein structure prediction with all-atom generation capabilities by integrating three proven architectural components: a modern ESMC-style encoder, ESMFold's evoformer inspired folding trunk, and AlphaFold3's all atom diffusion decoder. ESMFold achieves fast backbone-only predictions without MSAs (multiple sequence alignments) whereas AlphaFold3 generates complete atomic structures with ligands and PTMs, but requires MSAs as inputs which are expensive to compute. Our project ForgeFold bridges this gap by training a compact encoder from scratch (24 transformer layers × 768D) on 70M sequences with an MLM (masked language modeling pretraining task), connecting it to a 24-block, 512D folding trunk for geometric reasoning, and using a geometry-aware diffusion transformer (DiT) decoder for all-atom coordinate generation 12 layers x 512D. By combining the speed of single-sequence methods with the completeness of all-atom models in a tractable model, we aim to enable fast, general-purpose structure prediction for novel proteins, designed sequences, and protein-ligand complexes. We will train and evaluate ForgeFold against ESMFold (backbone-only baseline), AlphaFold2, OpenFold, RF3, Boltz, and AlphaFold3 on CASP and CAMEO structure prediction benchmarks to quantify accuracy and compare to existing state of the art models.


Background:


Protein structure prediction faces a fundamental trade-off between speed, generality, and completeness. MSA-based methods like AlphaFold2 achieve excellent accuracy when homologs exist but require minutes of sequence search and fail for orphan or designed proteins. Single-sequence methods like ESMFold are fast and work without MSAs, but only predict backbone atoms. All-atom methods like AlphaFold3 can generate complete structures including ligands and PTMs, but still rely on MSAs or complex preprocessing.

Our work bridges this gap by combining the architectural strengths of recent breakthroughs:

ESMC (2024) showed that modern transformer architectures enable high performance with much fewer parameters while capturing evolutionary patterns from sequences alone, without requiring MSAs
ESMFold (2022) demonstrated that large language models plus folding trunk modules (48 blocks) can predict backbone structures competitively with AlphaFold2 without the need for MSAs
AlphaFold3 (2024) introduced diffusion-based all-atom generation using atom-level attention, enabling prediction of proteins, nucleic acids, ligands, and post-translational modifications
RoseTTAFold3 and Boltz (2024) Open source implementations of AlphaFold 3
Project Sketch and Time Estimates:

Two-Stage Training Strategy

Stage A: Encoder Pre-training (~1-2 weeks)

Train compact encoder from scratch on 70M UniRef50 sequences
Objective: Progressive Masked language modeling (5% to 15% masking rate)
Target: Perplexity <8.0 (BERT-Base scale model)
Output: Pre-trained encoder that captures evolutionary and structural patterns
Stage B: Structure Prediction Training (~2-3 weeks)

Jointly train folding trunk + diffusion decoder
Frozen encoder
Dataset: ~150K PDB structures (proteins only )
Objective: adapted from AlphaFold2 and ESMFold
Target: Backbone RMSD <3.5Å, competitive all-atom accuracy
Reduced compute: Smaller models train 2-3× faster with similar GPU memory
Questions for Course Staff:

Q1: Computational Resources What GPU resources are available for this project? 

Q2: Dataset Access Are there recommended sources or pre-processed datasets for UniRef50 (65M sequences) and PDB structures

Q3: Evaluation Benchmarks Which benchmark sets should we prioritize for fair comparison: CASP14/15 targets (most cited), CAMEO continuous evaluation (realistic difficulty), or internal held-out PDB split (fastest to compute)? Should we run baselines ourselves or cite published numbers?

Q4: Project Scope Given a 5 week timeline and compact model, what would constitute a successful outcome: (a) minimum viable = encoder pre-trained + architecture implemented + preliminary results on proteins, (b) target = full pipeline trained + competitive backbone accuracy + demonstrated all-atom capability on simple complexes, or (c) stretch = all-atom capability validated on diverse molecule types (proteins, ligands, nucleic acids) with ablation studies? If compute or time constraints arise, should we prioritize the encoder training on a MLM task

Q5: Deliverable Format What format is expected for the course deliverable.

References:

ESMC (2024) - Lin, Z., et al. "Evolutionary-scale prediction of atomic-level protein structure with a language model."
ESMFold (2022) - Lin, Z., et al. "Language models of protein sequences predict structure." Science 379.6637 (2023).
AlphaFold3 (2024) - Abramson, J., et al. "Accurate structure prediction of biomolecular interactions with AlphaFold 3." Nature 630.8016 (2024).
RoseTTAFold3 (2024) - Krishna, R., et al. "Generalized biomolecular modeling and design with RoseTTAFold All-Atom."
Boltz-1 (2024) - Wohlwend, J., et al. "Boltz-1: Democratizing biomolecular interaction modeling."



## Next Steps
1. Train ESM2 module
3. Train entire thing
4. Inference


## Terms that I'm gonna google later:
- atom37 vs atom14 as protein structure formats.
- [BioEmu](https://www.nature.com/articles/s41592-025-02874-1) as a method for looking at polymer protein structures
- [ESM model](https://huggingface.co/docs/transformers/en/model_doc/esm).
- SwiGlu activation function. [paper](https://arxiv.org/pdf/2002.05202), [blog with easier to read implementation details](https://azizbelaweid.substack.com/p/what-is-swiglu-how-to-implement-it).
- [Weights and Biases](https://wandb.ai/site/) for babying model during extremely long training times.
- [Grouped Query Attention](https://arxiv.org/pdf/2305.13245): not every head has a different key, value pair. Queries are still unique. Reduces number of params. Multi Query Attention just uses a single key, value pair and different queries.
-  [Multihead Latent Attention](https://arxiv.org/abs/2502.07864) and a nice [blog post](https://planetbanatt.net/articles/mla.html). Here we force all key, value pairs to be embedded into a low rank v space using 2 smaller matrices. Then our learnable space is smaller and we require an extra matmul each time, but we store way less data and can potentially have better results.

## Quetions
- How does ESM fold do protein tokenization?
- What open models and weights are there for ESM2
- What is the AlphaFold3 diffusion model structure? Is it opensource or openweight? How was it trained?
