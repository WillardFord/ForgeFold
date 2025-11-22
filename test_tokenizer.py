from tokenizer import ESMCTokenizer

def test_tokenizer():
    tok = ESMCTokenizer()
    fasta_path = "/Users/liyuxin/Documents/zotero/PhD/Course/final_project/ForgeFold/uniparc30_sample_100.fasta"

    sequences = tok.read_fasta(fasta_path)
    print(f"Loaded {len(sequences)} sequences from FASTA")

    for i, seq in enumerate(sequences[:3]):
        ids = tok.encode(seq)
        decoded = tok.decode(ids)
        print(f"\nSequence {i+1}:")
        print(f"  Original length: {len(seq)}")
        print(f"  Encoded length: {len(ids)}")
        print(f"  Match: {decoded == seq}")
        assert decoded == seq, f"Tokenizer test failed for sequence {i+1}!"

    print("\nTokenizer test passed!")

if __name__ == "__main__":
    test_tokenizer()
