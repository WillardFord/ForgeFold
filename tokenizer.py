class ESMCTokenizer:
    """
    Minimal tokenizer for ESM C family (e.g., esmc-300m-2024-12).
    Matches the official Alphabet tokenizer behavior.
    """

    def __init__(self):
        # --- Special tokens (same as ESM2/ESMC) ---
        self.special_tokens = ["<pad>", "<cls>", "<eos>", "<unk>", "<mask>"]

        # --- 20 canonical amino acids ---
        self.amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

        # --- Extended amino acids / placeholders ---
        # From esm/tokenization/alphabet.py:
        # U = selenocysteine
        # O = pyrrolysine
        # B/Z = ambiguous
        # X = unknown
        # '-' = gap
        self.extended = ["U", "O", "B", "Z", "X", "-"]

        # Final vocab
        self.vocab = self.special_tokens + self.amino_acids + self.extended

        # maps
        self.tok_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_tok = {i: tok for tok, i in self.tok_to_id.items()}

        # helpful ids
        self.pad_idx = self.tok_to_id["<pad>"]
        self.cls_idx = self.tok_to_id["<cls>"]
        self.eos_idx = self.tok_to_id["<eos>"]
        self.unk_idx = self.tok_to_id["<unk>"]
        self.mask_idx = self.tok_to_id["<mask>"]

    # -------------------------
    # Encoding: AA string → token ids
    # -------------------------
    def encode(self, seq: str):
        seq_ids = []
        for aa in seq:
            seq_ids.append(self.tok_to_id.get(aa, self.unk_idx))

        return [self.cls_idx] + seq_ids + [self.eos_idx]

    # -------------------------
    # Decoding: ids → AA string
    # -------------------------
    def decode(self, ids):
        toks = []
        for i in ids:
            tok = self.id_to_tok.get(i, "<unk>")
            if tok not in self.special_tokens:
                toks.append(tok)
        return "".join(toks)

    # optional utility
    def __len__(self):
        return len(self.vocab)

    # -------------------------
    # Read FASTA file
    # -------------------------
    def read_fasta(self, file_path: str):
        samples = []
        current_seq = []

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        samples.append(''.join(current_seq))
                        current_seq = []
                else:
                    current_seq.append(line)

            if current_seq:
                samples.append(''.join(current_seq))

        return samples
