LIKE_DISLIKE_EXTRACTION = """
Given a dialogue history between the User and the System, extract movie-related preferences that reflect what the user is currently seeking.
Classify the extracted preferences into two aspects: Like and Dislike.
If there is nothing to mention for an aspect, write "None." under the corresponding tag.

Dialogue history: {dialog}
Response:
[Like] {{keyphrases or descriptions separated by commas}}
[Dislike] {{keyphrases or descriptions separated by commas}}
"""


LIKE_DISLIKE_AUGMENTATION = """
You are an advanced user preference profile generator.
Based on the conversation and the extracted like/dislike preferences, infer and expand the user's potential movie preferences.
Augment key phrases related to the user's likes and dislikes, including preferences that may not have been explicitly stated.
If no explicit user preferences are provided, infer them from the conversation.
Do not include unrelated information. Only state the user's movie preferences.

User preferences: {pref_extracted}
Conversation: {dialog}
Response:
[Like] {{expanded keyphrases describing the user's likes}}
[Dislike] {{expanded keyphrases describing the user's dislikes}}
"""

MOOD_EXTRACTION = """
You are a movie mood and atmosphere analyst. 
From the conversation, extract the **ABSTRACT MOOD, EMOTIONAL TONE, and ATMOSPHERIC QUALITY** the user seems to be seeking.
1. Extract the **emotional atmosphere, viewing mood, and tonal qualities** implied by the conversation.
2. Use abstract descriptive phrases like: "edge-of-seat suspense", "nostalgic comfort", "visually immersive", "emotionally devastating", "darkly atmospheric", "feel-good warmth", "intellectual stimulation", "adrenaline-fueled intensity", "melancholic beauty", "quirky offbeat humor", "gritty realism".
3. DO NOT use genre labels (action, comedy, thriller, romance, horror, drama, sci-fi, animation, documentary, mystery, etc.) — those belong to a separate signal.
4. DO NOT include movie titles, actor names, director names, franchise names, or any named entities.
5. DO NOT include behavioral descriptions (rejected X, accepted Y, watching with someone). Focus ONLY on mood/tone/atmosphere.
6. Output 2-4 comma-separated mood phrases.
7. For short conversations, infer a general mood from whatever context is available (e.g., "lighthearted casual entertainment").
8. Output ONLY comma-separated mood phrases. No sentences, no numbering, no prefixes.

Dialogue history: {dialog}
Response:
"""

VIEWING_CONTEXT_PROMPT = """
You are a conversation dynamics analyst. 
Extract the user's **REACTIONS, CONTEXT, and CONSTRAINTS** from the conversation as comma-separated keywords.

1. Extract the **specific reactions to suggestions** (accepted/rejected and why), **situational context** (watching with whom, mood, viewing situation), **hard constraints**, and **conversation direction**.
2. STRICTLY DO NOT include any movie titles, actor names, director names, franchise names, or any named entities. Describe reactions WITHOUT naming specific movies (e.g., "enthusiastically accepted system recommendation" NOT "enthusiastically accepted Parasite").
3. DO NOT restate genre preferences or movie attributes — focus on WHAT IS HAPPENING in the conversation.
4. Focus on behavioral signals: enthusiasm, hesitation, acceptance, rejection, curiosity, follow-up questions, social dynamics.
5. For short conversations with few reactions, describe the conversation stage and openness level (e.g., "initial exploration, open to suggestions").
6. Output 2-7 comma-separated keyword phrases depending on available info. Nothing else — no prefix, no sentences, no numbering.

Dialogue history: {dialog}
Response:
"""