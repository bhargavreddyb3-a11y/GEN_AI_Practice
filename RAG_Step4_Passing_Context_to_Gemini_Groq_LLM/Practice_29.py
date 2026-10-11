from Practice_28 import collection, store_document, rag_answer

third_doc = """Space Missions and Exploration

Robotic probes have visited every planet in the Solar System. Rovers on Mars drive across the surface, drill
into rocks, and send photographs back to Earth.

Human spaceflight requires life support, radiation shielding, and careful training. Astronauts on the
International Space Station orbit the Earth roughly every ninety minutes and run experiments in microgravity.
"""

store_document("space_missions.txt", third_doc)
print("Total chunks now:", collection.count())

for q in ['How do rovers explore Mars?', 'How long does the ISS take to orbit the Earth?', 'What is the best recipe for chocolate cake?']:
    print("\nQuestion:", q)
    print(rag_answer(q))
