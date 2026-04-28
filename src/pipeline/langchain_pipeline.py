"""
LangChain + Gemini pipeline for SignBridge.

Takes a buffer of recognized ASL signs (letters and words) and returns
a fluent English sentence using Gemini 1.5 Flash via Vertex AI.

Usage:
    from src.pipeline.langchain_pipeline import SignBridgePipeline

    pipeline = SignBridgePipeline()
    result = pipeline.translate(["HELLO", "M", "Y", "NAME", "IS", "N", "O", "S", "H"])
    print(result["sentence"])   # "Hello, my name is Nosh."
    print(result["latency_ms"]) # e.g. 512.3
"""

import os
import time
import logging
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()  # loads .env from project root

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert ASL (American Sign Language) interpreter assistant.
You will receive a sequence of recognized ASL signs. These signs may be:
- Full words (e.g. HELLO, HELP, THANK)
- Individual fingerspelled letters (e.g. N, O, S, H)

Your job is to reconstruct the most natural, grammatically correct English sentence from this sequence.

Rules:
1. Consecutive letters should be combined into a word (e.g. N-O-S-H → "Nosh")
2. ASL grammar differs from English — reorder words as needed for natural English
3. Add appropriate punctuation
4. If the sequence is unclear, make your best reasonable interpretation
5. Return ONLY the reconstructed sentence — no explanations, no preamble
"""

USER_TEMPLATE = "Signs: {signs}\nSentence:"


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class SignBridgePipeline:
    """
    LangChain + Gemini 1.5 Flash pipeline for sign-to-sentence translation.

    Args:
        project_id:      GCP project ID (defaults to GOOGLE_CLOUD_PROJECT env var)
        location:        Vertex AI region (default: us-central1)
        model_name:      Gemini model to use (default: gemini-1.5-flash)
        max_retries:     Number of retries on API failure (default: 3)
        context_window:  Max number of signs to send to LLM (default: 10)
    """

    def __init__(
        self,
        project_id: str = None,
        location: str = "us-central1",
        model_name: str = "gemini-2.5-flash",
        max_retries: int = 3,
        context_window: int = 10,
    ):
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT", "signbridge-prod")
        self.location = location
        self.model_name = model_name
        self.max_retries = max_retries
        self.context_window = context_window

        self._llm = None
        self._chain = None
        self._init_chain()

    def _init_chain(self):
        """Initialise LangChain chain with Gemini via Google AI API."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            self._llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=0.3,
                max_output_tokens=128,
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", USER_TEMPLATE),
            ])

            self._chain = prompt | self._llm | StrOutputParser()
            logger.info(f"LangChain pipeline initialised — model: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to initialise LangChain pipeline: {e}")
            raise

    def translate(self, signs: List[str]) -> Dict:
        """
        Translate a list of ASL sign tokens into a fluent English sentence.

        Args:
            signs: List of sign strings e.g. ["HELLO", "M", "Y", "NAME"]

        Returns:
            dict with keys:
                sentence    (str)   — translated sentence
                input_signs (list)  — signs that were sent (after context window trim)
                latency_ms  (float) — API round-trip time in milliseconds
                success     (bool)  — False if all retries failed
                error       (str)   — error message if success=False, else None
        """
        # Trim to context window
        signs_trimmed = signs[-self.context_window:]
        signs_str = " ".join(signs_trimmed)

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                t0 = time.time()
                sentence = self._chain.invoke({"signs": signs_str})
                latency_ms = (time.time() - t0) * 1000

                sentence = sentence.strip()
                logger.info(f"Translated in {latency_ms:.0f}ms: {signs_trimmed} → '{sentence}'")

                return {
                    "sentence": sentence,
                    "input_signs": signs_trimmed,
                    "latency_ms": round(latency_ms, 1),
                    "success": True,
                    "error": None,
                }

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)   # exponential backoff: 2s, 4s

        logger.error(f"All {self.max_retries} attempts failed. Last error: {last_error}")
        return {
            "sentence": "[Translation failed — please try again]",
            "input_signs": signs_trimmed,
            "latency_ms": 0.0,
            "success": False,
            "error": last_error,
        }

    def translate_batch(self, sequences: List[List[str]]) -> List[Dict]:
        """Translate multiple sign sequences. Used for LLM eval split testing."""
        return [self.translate(seq) for seq in sequences]
