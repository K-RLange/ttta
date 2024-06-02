from src.ttta.methods.sense_representation import SenseRepresentation

if __name__ == '__main__':
    sense_embedding = SenseRepresentation(
        target_word='bank'
    ).infer_representation()