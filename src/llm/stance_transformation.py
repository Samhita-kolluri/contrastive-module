class StanceTransformer:
    def generate_opposing_stances(self, text):
        return [text, f"Not {text}", f"Against {text}", f"Alternative to {text}"]