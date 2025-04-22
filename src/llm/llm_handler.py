import logging
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class LLMHandler:
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.model_name = "tiiuae/falcon-rw-1b"  # works fast on CPU plus a fined-tune model 
        self.num_return_sequences = config["llm"].get("num_return_sequences", 4)
        self.max_tokens = config["llm"].get("max_tokens", 256)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.logger.info(f"Loading LLM model: {self.model_name}")
        
        # Load the model and tokenizer
        try:
            # Try loading the tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.logger.info("Tokenizer loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {e}")
            raise ValueError(f"Model tokenizer for {self.model_name} could not be loaded.") 

        try:
            # loading the model
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
            self.logger.info("LLM model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load LLM: {e}")
            raise ValueError(f"Model {self.model_name} could not be loaded.")
        
        # Initialize text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        self.logger.info("LLM successfully loaded and ready.")

    def generate_contrastive_docs(self, input_text: str, num_docs: int = 4) -> List[str]:
        """
        Generate contrastive statements using few-shot prompting.
        """
        prompt = self._build_prompt(input_text)
        self.logger.info(f"Generating {num_docs} contrastive documents for: '{input_text}'")

        try:
            outputs = self.generator(
                prompt,
                max_length=len(self.tokenizer(prompt)["input_ids"]) + self.max_tokens,
                num_return_sequences=num_docs,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                eos_token_id=self.tokenizer.eos_token_id
            )
        except Exception as e:
            self.logger.error(f"Error during text generation: {e}")
            raise e

        results = [self._postprocess(out["generated_text"], prompt) for out in outputs]
        return [r for r in results if r.strip() != ""]

    def _build_prompt(self, input_text: str) -> str:
        """
        Create a prompt with few-shot examples + user query.
        """
        examples = [
            ("Climate change is primarily driven by human activities.",
             "Climate change is a natural cycle unrelated to human actions."),

            ("AI will revolutionize healthcare in the next decade.",
             "AI's impact on healthcare is overstated and will be marginal."),

            ("Social media promotes democratic participation.",
             "Social media deepens echo chambers and undermines true discourse.")
        ]

        prompt = "Below are claims followed by contrasting viewpoints:\n\n"
        for original, contrast in examples:
            prompt += f"Claim: {original}\nContrast: {contrast}\n\n"

        prompt += f"Claim: {input_text}\nContrast:"
        return prompt

    def _postprocess(self, output: str, prompt: str) -> str:
        """
        Remove prompt and cleanup formatting.
        """
        gen = output[len(prompt):].strip()
        if "\n" in gen:
            gen = gen.split("\n")[0].strip()
        return gen
