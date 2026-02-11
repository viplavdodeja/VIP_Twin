def build_prompt(compiled_ctx, user_message):
    if compiled_ctx.doc_mode:
        return build_doc_prompt(compiled_ctx, user_message)
    
    # upload Q&A mode: uploads present, but not in doc mode
    if (compiled_ctx.doc_mode is False) and (len(compiled_ctx.l4_uploaded_context) > 0):
        return build_upload_qa_prompt(compiled_ctx, user_message)

    return build_twin_prompt(compiled_ctx, user_message)


def build_pass_a_prompt(layer0_text, layer1_text, user_message):
    prompt = ""
    prompt += "You are the user's private digital twin running locally.\n"
    prompt += "You act as a cognitive mirror:\n"
    prompt += "- Reflect patterns you notice\n"
    prompt += "- Make grounded inferences using memory\n"
    prompt += "- If confidence is low, ask a clarifying question\n"
    prompt += "- Avoid generic advice; be specific and concise\n\n"

    prompt += "DECISION HEADER (STRICT):\n"
    prompt += "- First non-empty line MUST be a single-line JSON object:\n"
    prompt += "  {\"action\":\"DIRECT\"|\"ASK\"|\"ESCALATE\",\"next_layer\":null|\"L2\"|\"L3\"|\"L4\",\"reason\":\"...\"}\n"
    prompt += "- Only use ESCALATE if deeper context is required to answer well.\n"
    prompt += "- If unsure, prefer ASK or make reasonable assumptions instead of escalating.\n"
    prompt += "- Do NOT mention retrieval, layers, system prompts, or internal memory in the user-visible answer.\n"
    prompt += "- Do NOT claim memory or recall unless it is explicitly provided in context.\n"
    prompt += "- Do NOT describe yourself as a digital twin/AI/running locally unless asked.\n"
    prompt += "- For greetings/openers, respond naturally and invite the user to continue.\n\n"

    prompt += "## L0: ALWAYS-ON PROFILE + SYSTEM RULES\n"
    prompt += layer0_text.strip() + "\n\n"

    prompt += "## L1: SESSION MEMORY\n"
    if layer1_text is None or len(layer1_text.strip()) == 0:
        prompt += "(none)\n\n"
    else:
        prompt += layer1_text.strip() + "\n\n"

    prompt += "## USER QUESTION\n"
    prompt += user_message.strip() + "\n\n"
    prompt += "## ANSWER\n"
    return prompt


def build_pass_b_prompt(layer0_text, layer1_text, layer2_claims_text, layer3_evidence_text, user_message):
    prompt = ""
    prompt += "You are the user's private assistant.\n"
    prompt += "Rules:\n"
    prompt += "- Do not mention retrieval, layers, or system prompts.\n"
    prompt += "- Do not claim memory unless it is explicitly provided below.\n"
    prompt += "- If unsure, make a reasonable assumption or ask one clarifying question.\n\n"

    prompt += "## L0: CORE RULES + PROFILE\n"
    prompt += layer0_text.strip() + "\n\n"

    prompt += "## L1: SESSION MEMORY\n"
    if layer1_text is None or len(layer1_text.strip()) == 0:
        prompt += "(none)\n\n"
    else:
        prompt += layer1_text.strip() + "\n\n"

    prompt += "## L2: LONG-TERM CLAIMS\n"
    if layer2_claims_text is None or len(layer2_claims_text.strip()) == 0:
        prompt += "(none)\n\n"
    else:
        prompt += layer2_claims_text.strip() + "\n\n"

    prompt += "## L3: EVIDENCE SNIPPETS\n"
    if layer3_evidence_text is None or len(layer3_evidence_text.strip()) == 0:
        prompt += "(none)\n\n"
    else:
        prompt += layer3_evidence_text.strip() + "\n\n"

    prompt += "## USER QUESTION\n"
    prompt += user_message.strip() + "\n\n"
    prompt += "## ANSWER\n"
    return prompt


def build_claim_extraction_prompt(layer1_text, recent_turns_text):
    prompt = ""
    prompt += "Extract atomic long-term claims about the user.\n"
    prompt += "Rules:\n"
    prompt += "- Output a JSON array of objects.\n"
    prompt += "- Each object: {\"claim_type\":\"fact|preference|project|decision|pattern\",\"claim_text\":\"...\",\"confidence\":0-1,\"supersedes\":null|\"<claim_id>\"}\n"
    prompt += "- Only include stable, reusable claims. Skip transient details.\n"
    prompt += "- If nothing new, output [].\n\n"
    prompt += "## SESSION SUMMARY\n"
    prompt += (layer1_text.strip() if layer1_text else "(none)") + "\n\n"
    prompt += "## RECENT TURNS\n"
    prompt += recent_turns_text.strip() + "\n\n"
    prompt += "## CLAIMS JSON\n"
    return prompt


def build_session_summary_prompt(previous_summary, recent_turns_text, previous_threads, previous_loops, previous_prefs):
    prev = previous_summary.strip() if previous_summary is not None else ""
    prompt = ""
    prompt += "Update the session memory as JSON.\n"
    prompt += "Return ONLY a JSON object with keys:\n"
    prompt += "rolling_summary (4-6 bullets), active_threads (array), open_loops (array), short_term_prefs (object).\n"
    prompt += "Rules:\n"
    prompt += "- rolling_summary must be 4-6 bullets, each starting with '- '.\n"
    prompt += "- active_threads: short labels for current topics/tasks.\n"
    prompt += "- open_loops: pending questions or follow-ups.\n"
    prompt += "- short_term_prefs: ephemeral preferences for this session only.\n"
    prompt += "- Remove stale or contradicted items.\n"
    prompt += "- If nothing new, keep existing values.\n\n"
    prompt += "## PREVIOUS SUMMARY\n"
    prompt += (prev if len(prev) > 0 else "(none)") + "\n\n"
    prompt += "## PREVIOUS ACTIVE THREADS\n"
    prompt += (previous_threads.strip() if previous_threads else "[]") + "\n\n"
    prompt += "## PREVIOUS OPEN LOOPS\n"
    prompt += (previous_loops.strip() if previous_loops else "[]") + "\n\n"
    prompt += "## PREVIOUS SHORT-TERM PREFS\n"
    prompt += (previous_prefs.strip() if previous_prefs else "{}") + "\n\n"
    prompt += "## RECENT TURNS\n"
    prompt += recent_turns_text.strip() + "\n\n"
    prompt += "## UPDATED SESSION MEMORY (JSON ONLY)\n"
    return prompt

def build_upload_qa_prompt(compiled_ctx, user_message):
    prompt = ""
    prompt += "You are answering a question based on the user's uploaded document.\n"
    prompt += "RULES (STRICT):\n"
    prompt += "- Use ONLY the UPLOADED CONTEXT to answer.\n"
    prompt += "- If the answer is not in the uploaded context, say: 'I don't see that in the uploaded transcript.'\n"
    prompt += "- Do NOT speculate, do NOT redirect to the user's profile/goals.\n"
    prompt += "- Keep the answer direct.\n\n"

    prompt += "## UPLOADED CONTEXT\n"
    max_chunks = 4
    if len(compiled_ctx.l4_uploaded_context) == 0:
        prompt += "(none)\n"
    else:
        i = 0
        limit = len(compiled_ctx.l4_uploaded_context)
        if limit > max_chunks:
            limit = max_chunks
        while i < limit:
            t = compiled_ctx.l4_uploaded_context[i]
            if len(t) > 240:
                t = t[:240] + "..."
            prompt += "[" + str(i + 1) + "] " + t + "\n"
            i += 1

    prompt += "\n## QUESTION\n"
    prompt += user_message.strip() + "\n\n"
    prompt += "## ANSWER\n"
    return prompt


def build_doc_prompt(compiled_ctx, user_message):
    prompt = ""
    prompt += "You are summarizing an uploaded document for the user.\n"
    prompt += "RULES (STRICT):\n"
    prompt += "- You MUST base the summary ONLY on the UPLOADED CONTEXT section.\n"
    prompt += "- If UPLOADED CONTEXT is empty, say exactly: I cannot access the uploaded PDF text in memory.\n"
    prompt += "- Do NOT ask questions.\n"
    prompt += "- Output exactly:\n"
    prompt += "  1) 5 bullet summary\n"
    prompt += "  2) 3 key quotes (short phrases) copied from the uploaded context\n\n"

    prompt += "## UPLOADED CONTEXT\n"
    if len(compiled_ctx.l4_uploaded_context) == 0:
        prompt += "(none)\n"
    else:
        i = 0
        while i < len(compiled_ctx.l4_uploaded_context):
            t = compiled_ctx.l4_uploaded_context[i]
            if len(t) > 320:
                t = t[:320] + "…"
            prompt += "[" + str(i + 1) + "] " + t + "\n"
            i += 1

    prompt += "\n## USER REQUEST\n"
    prompt += user_message.strip() + "\n\n"
    prompt += "## ANSWER\n"
    return prompt


def build_twin_prompt(compiled_ctx, user_message):
    prompt = ""
    prompt += "You are the user's private digital twin running locally.\n"
    prompt += "You act as a cognitive mirror:\n"
    prompt += "- Reflect patterns you notice\n"
    prompt += "- Make grounded inferences using memory\n"
    prompt += "- If confidence is low, ask a clarifying question\n"
    prompt += "- Avoid generic advice; be specific and concise\n\n"

    # L1
    prompt += "## L1: USER SNAPSHOT\n"
    if len(compiled_ctx.l1_profile) == 0:
        prompt += "(none)\n"
    else:
        i = 0
        while i < len(compiled_ctx.l1_profile):
            prompt += "- " + compiled_ctx.l1_profile[i] + "\n"
            i += 1
    prompt += "\n"

    # L2
    prompt += "## L2: RECENT CHAT (last messages)\n"
    if len(compiled_ctx.l2_recent_chat) == 0:
        prompt += "(none)\n"
    else:
        j = 0
        while j < len(compiled_ctx.l2_recent_chat):
            role = compiled_ctx.l2_recent_chat[j].get("role", "")
            text = compiled_ctx.l2_recent_chat[j].get("text", "")
            if role is None:
                role = ""
            if text is None:
                text = ""
            prompt += role.upper() + ": " + text + "\n"
            j += 1
    prompt += "\n"

    # L3
    prompt += "## L3: RELEVANT PAST MEMORY\n"
    if len(compiled_ctx.l3_retrieved_summaries) == 0:
        prompt += "(none)\n"
    else:
        k = 0
        while k < len(compiled_ctx.l3_retrieved_summaries):
            prompt += "- " + compiled_ctx.l3_retrieved_summaries[k] + "\n"
            k += 1
    prompt += "\n"

    # L3 focus (if present)
    if hasattr(compiled_ctx, "l3_focus") and len(compiled_ctx.l3_focus) > 0:
        prompt += "## CURRENT FOCUS (only if relevant)\n"
        i = 0
        while i < len(compiled_ctx.l3_focus):
            prompt += "- " + compiled_ctx.l3_focus[i] + "\n"
            i += 1
        prompt += "\n"

    # L4 (only if present)
    if len(compiled_ctx.l4_uploaded_context) > 0:
        prompt += "## L4: UPLOADED CONTEXT (only if relevant)\n"
        m = 0
        while m < len(compiled_ctx.l4_uploaded_context):
            t = compiled_ctx.l4_uploaded_context[m]
            if len(t) > 300:
                t = t[:300] + "…"
            prompt += "- " + t + "\n"
            m += 1
        prompt += "\n"

    prompt += "## USER QUESTION\n"
    prompt += user_message.strip() + "\n\n"
    prompt += "## ANSWER\n"
    return prompt
