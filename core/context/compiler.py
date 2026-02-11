class CompiledContext:
    def __init__(self):
        self.l1_profile = []
        self.l2_recent_chat = []
        self.l3_retrieved_summaries = []
        self.l3_focus = []
        self.l4_uploaded_context = []

        self.doc_mode = False

        self.debug = {}


def compile_context(service, user_id, chat_id, user_message, route):
    ctx = CompiledContext()
    ctx.doc_mode = route.doc_mode
    fast_path = hasattr(route, "fast_path") and route.fast_path

    # L1 profile (short snapshot)
    if route.needs_profile and (fast_path is False):
        profile_texts = service.get_identity_profile_texts(user_id)
        if len(profile_texts) > 3:
            profile_texts = profile_texts[:3]
        ctx.l1_profile = service.enforce_context_budget(profile_texts, max_chars=1200)

    # L2 recent chat window (last N messages)
    if route.needs_recent_chat and (fast_path is False) and (route.doc_mode is False):
        ctx.l2_recent_chat = service.get_chat_messages(user_id, chat_id, limit=10)

    # L3 focus profile (conditional)
    if hasattr(route, "needs_focus") and route.needs_focus and (fast_path is False):
        ctx.l3_focus = service.get_focus_profile_texts(user_id)

    # L3 retrieved long-term summaries (conditional)
    if route.summary_top_k > 0 and (fast_path is False):
        ctx.l3_retrieved_summaries = service.retrieve_relevant_summaries(
            user_id=user_id,
            chat_id=chat_id,
            query_text=user_message,
            top_k=route.summary_top_k
        )
        ctx.l3_retrieved_summaries = service.enforce_context_budget(
            ctx.l3_retrieved_summaries,
            max_chars=1500
        )

    # L4 uploaded retrieval (conditional)
    if route.upload_top_k > 0:
        ctx.l4_uploaded_context = service.retrieve_relevant_uploaded_context(
            user_id=user_id,
            query_text=user_message,
            top_k=route.upload_top_k
        )
        upload_budget = 2000
        if route.doc_mode:
            upload_budget = 5000
        ctx.l4_uploaded_context = service.enforce_context_budget(
            ctx.l4_uploaded_context,
            max_chars=upload_budget
        )

    # debug
    ctx.debug = {
        "profile_count": len(ctx.l1_profile),
        "recent_chat_count": len(ctx.l2_recent_chat),
        "retrieved_summary_count": len(ctx.l3_retrieved_summaries),
        "retrieved_upload_count": len(ctx.l4_uploaded_context),
        "doc_mode": ctx.doc_mode,
        "upload_top_k": route.upload_top_k,
        "summary_top_k": route.summary_top_k,
        "focus_count": len(ctx.l3_focus),
        "fast_path": fast_path,
    }

    return ctx
