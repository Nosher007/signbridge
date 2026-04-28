"""
Day 5 test — LangChain + Gemini pipeline.

Run on GCP VM:
    python tests/test_langchain_pipeline.py

All 10 test cases must pass and print quality scores before this task is done.
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline.langchain_pipeline import SignBridgePipeline

# ---------------------------------------------------------------------------
# 10 test sequences covering common patterns
# ---------------------------------------------------------------------------
TEST_CASES = [
    # (input signs, description)
    (["HELLO", "MY", "NAME", "IS", "N", "O", "S", "H"],   "intro with fingerspelling"),
    (["HELP", "M", "E"],                                    "word + fingerspelled letters"),
    (["THANK", "YOU"],                                       "simple 2-word phrase"),
    (["WHERE", "IS", "THE", "BATHROOM"],                    "question"),
    (["I", "LOVE", "YOU"],                                   "common phrase"),
    (["MY", "NAME", "IS", "A", "Y", "U", "S", "H"],        "intro with full name spelling"),
    (["NICE", "TO", "MEET", "YOU"],                         "greeting"),
    (["CAN", "YOU", "HELP", "ME"],                          "request"),
    (["GOOD", "MORNING", "HOW", "ARE", "YOU"],              "morning greeting"),
    (["I", "AM", "LEARNING", "A", "S", "L"],                "statement with acronym"),
]

def score_quality(sentence: str) -> int:
    """Simple 1-5 quality score based on heuristics."""
    if not sentence or "[Translation failed" in sentence:
        return 1
    words = sentence.split()
    if len(words) < 2:
        return 2
    has_punct = sentence[-1] in ".!?"
    reasonable_length = 3 <= len(words) <= 20
    score = 3
    if has_punct:
        score += 1
    if reasonable_length:
        score += 1
    return min(score, 5)

def main():
    print("=" * 60)
    print("SignBridge — LangChain + Gemini Pipeline Test")
    print("=" * 60)

    pipeline = SignBridgePipeline()
    print(f"Pipeline ready — model: {pipeline.model_name}\n")

    results = []
    total_latency = 0.0
    failures = 0

    print(f"{'#':<3} {'Input Signs':<45} {'Output Sentence':<40} {'Q':>2} {'ms':>6}")
    print("-" * 100)

    for i, (signs, desc) in enumerate(TEST_CASES, 1):
        result = pipeline.translate(signs)

        if not result["success"]:
            failures += 1

        quality = score_quality(result["sentence"])
        total_latency += result["latency_ms"]
        results.append({**result, "quality": quality, "description": desc})

        signs_str = " ".join(signs)[:43]
        sentence_str = result["sentence"][:38]
        print(f"{i:<3} {signs_str:<45} {sentence_str:<40} {quality:>2} {result['latency_ms']:>6.0f}")

    print("-" * 100)

    avg_quality = sum(r["quality"] for r in results) / len(results)
    avg_latency = total_latency / len(results)
    success_rate = (len(TEST_CASES) - failures) / len(TEST_CASES) * 100

    print(f"\nSummary:")
    print(f"  Test cases      : {len(TEST_CASES)}")
    print(f"  Success rate    : {success_rate:.0f}%")
    print(f"  API failures    : {failures}")
    print(f"  Avg quality     : {avg_quality:.1f} / 5")
    print(f"  Avg latency     : {avg_latency:.0f} ms")

    print("\nDetailed results:")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r['description']}")
        print(f"       Input  : {' '.join(r['input_signs'])}")
        print(f"       Output : {r['sentence']}")
        print(f"       Quality: {r['quality']}/5  |  Latency: {r['latency_ms']}ms  |  Success: {r['success']}")

    # Assertions
    assert failures == 0, f"FAIL: {failures} API failures — check Vertex AI auth"
    assert avg_quality >= 3.0, f"FAIL: avg quality {avg_quality:.1f} < 3.0"
    assert success_rate == 100.0, f"FAIL: success rate {success_rate}% < 100%"

    print("\n✓ All tests passed — Day 5 pipeline ready")
    return results

if __name__ == "__main__":
    main()
