from contextvars import ContextVar

request_id_var = ContextVar("request_id", default=None)
