class ContextRoute:
    def __init__(self):
        self.doc_mode = False
        self.needs_upload = False
        self.needs_long_memory = False
        self.needs_profile = True
        self.needs_recent_chat = True
        self.needs_focus = False
        self.include_layer1 = False

        # retrieval sizes
        self.upload_top_k = 0
        self.summary_top_k = 0

def classify_route(user_message, has_prior_turns=False):
    route = ContextRoute()
    # Decision-only: include Layer 1 based on whether this session has prior turns.
    route.include_layer1 = True if has_prior_turns else False
    route.needs_recent_chat = False
    route.needs_long_memory = False
    route.needs_upload = False
    route.doc_mode = False
    route.upload_top_k = 0
    route.summary_top_k = 0
    return route
