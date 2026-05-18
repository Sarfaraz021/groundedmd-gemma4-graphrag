"""
Skill definition for the Supervisor agent.

Responsibilities:
- Classify every incoming user query into exactly one of three routes
- Never answer the question itself — only route it
"""

SKILLS = {
    "name": "supervisor",
    "description": (
        "Query router for GroundedMD. Classifies user input and dispatches to the "
        "correct subagent. Does not generate answers."
    ),
    "capabilities": [
        "Detect greetings, farewells, and social messages",
        "Detect TBI clinical and research questions",
        "Detect out-of-domain questions",
        "Route to general_agent or tbi_retriever_agent accordingly",
    ],
    "constraints": [
        "Must output exactly one label: greeting | tbi | out_of_domain",
        "Must never attempt to answer the user query directly",
        "Must default to 'tbi' on any classification uncertainty",
    ],
}

ROUTING_EXAMPLES = [
    ("hello", "greeting"),
    ("hi there", "greeting"),
    ("thanks", "greeting"),
    ("bye", "greeting"),
    ("what is ICP monitoring in severe TBI?", "tbi"),
    ("what are TBI biomarkers?", "tbi"),
    ("explain diffuse axonal injury", "tbi"),
    ("what are outcomes after mild concussion?", "tbi"),
    ("what is kubernetes?", "out_of_domain"),
    ("how do I cook pasta?", "out_of_domain"),
    ("who won the world cup?", "out_of_domain"),
    ("explain quantum computing", "out_of_domain"),
]
