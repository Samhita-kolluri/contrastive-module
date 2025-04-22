class PromptEngineer:
    def build_prompt(self, input_text, topics, stances):
        return f"Generate {len(topics)} opposing views to '{input_text}' on topics: {', '.join(topics)} with stances: {', '.join(stances)}."
