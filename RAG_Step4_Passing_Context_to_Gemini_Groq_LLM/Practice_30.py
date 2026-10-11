from Practice_28 import retrieve_context, build_user_prompt, ask_llm

question = "Why are space telescopes better than ground ones?"
prompt = build_user_prompt(question, retrieve_context(question))

for temp in (0.0, 0.9):
    print("=" * 18, f"temperature = {temp}", "=" * 18)
    for run in range(1, 3):
        print(f"\nRun {run}:\n{ask_llm(prompt, temperature=temp)}")
