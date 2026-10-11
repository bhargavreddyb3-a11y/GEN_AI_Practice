from Practice_28 import rag_answer

THREE_BULLETS = (
    "Answer using ONLY the provided context. If the answer is missing, reply exactly: "
    "\"I don't know based on the provided documents.\" "
    "Format: exactly 3 bullet points, each starting with '- ' and containing one short sentence. "
    "After the bullets add one final line: Sources: <file names>."
)

for q in ['What is the asteroid belt?', 'How do telescopes and probes help us study the Solar System?']:
    answer = rag_answer(q, system_prompt=THREE_BULLETS, show_sources=False)
    bullets = [l for l in answer.splitlines() if l.strip().startswith("- ")]
    print(f"\nQuestion: {q}\n{answer}\nBullet count: {len(bullets)} (expected 3)")
