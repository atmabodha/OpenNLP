class PromptTemplates:

    default = """Question: {query}

Relevant verses from ancient texts:
{verses}

Provide a response that:
1. Directly addresses the question using wisdom from the provided verses.
2. Explains the practical application of this wisdom.
3. Connects the teachings of the verses to the question.
4. Excludes verse citations in your response—these will be added automatically.
5. Presents the response in clear, concise bullet points.

Answer in a conversational yet philosophically accurate style."""

    philosophical = """Question: {query}

Ancient wisdom relevant to your philosophical inquiry:
{verses}

Provide a response that:
1. Analyzes the core philosophical themes in these verses.
2. Explores their deeper implications on consciousness, reality, and existence.
3. Connects these insights to universal truths and classical teachings.
4. Relates them to modern-day philosophical discussions.
5. Presents key takeaways in clear, structured bullet points.

Answer in a thoughtful yet accessible manner, maintaining philosophical depth while keeping it easy to understand."""

    practical = """Question: {query}

Practical wisdom from ancient texts:
{verses}

Provide a response that:
1. Extracts key practical lessons from these verses.
2. Offers specific, actionable guidance based on these teachings.
3. Explains how to apply this wisdom in daily life.
4. Addresses common challenges and solutions from an ancient wisdom perspective.
5. Presents advice in concise bullet points for clarity.

Answer in a direct and actionable manner, ensuring practical relevance while maintaining philosophical depth."""

    clarification = """It seems like your question might be about: {query}. 

To provide the best insights from the Bhagavad Gita and Yoga Sutras, please consider:
1. Rephrasing your question to focus on philosophical or spiritual aspects.
2. Asking about specific teachings or principles.
3. Seeking guidance on applying ancient wisdom to this topic.

Would you like to refine your question to explore the deeper wisdom in these texts?"""

    comparative = """Question: {query}

Relevant verses from ancient texts:
{verses}

Provide a comparative analysis that:
1. Highlights the similarities and differences in the teachings.
2. Explains the context in which these teachings are presented.
3. Shows how each perspective contributes to spiritual growth.
4. Offers practical takeaways for the user.
5. Presents insights in structured bullet points.

Answer in a balanced and insightful manner, ensuring clarity and depth."""

    storytelling = """Question: {query}

Wisdom from ancient texts:
{verses}

Provide a response that:
1. Illustrates the key teachings through a relevant story or analogy.
2. Captures the essence of the lesson in an engaging way.
3. Provides a moral takeaway aligned with the scriptures.
4. Connects the story to modern-day experiences.
5. Keeps the response concise and structured in bullet points.

Answer in a narrative yet insightful manner, making the wisdom accessible and memorable."""


    meditation = """Question: {query}

Meditative wisdom from ancient texts:
{verses}

Provide a response that:
1. Guides the user through a reflective or meditative process.
2. Suggests contemplation techniques based on these teachings.
3. Helps the user internalize and apply the wisdom.
4. Encourages self-awareness and spiritual growth.
5. Presents the response in a structured and calming manner.

Answer with a meditative and introspective tone, promoting deep reflection."""
