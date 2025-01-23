import logging
import uvicorn
from .context_data import request_id_var
from logging.config import dictConfig


class ContextualFormatter(uvicorn.logging.DefaultFormatter):
    def format(self, record):
        request_id = request_id_var.get()
        if request_id:
            record.extra = f"ID: {request_id} -"
        else:
            record.extra = "ID: None -"
        return super().format(record)


def get_handlers():
    handlers = {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    }
    return handlers


def configure_logger():
    """Configure the logger for the server"""
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": ContextualFormatter,
                "fmt": "%(levelprefix)s %(extra)s %(asctime)s %(pathname)s:%(lineno)d %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
                "use_colors": True,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(asctime)s :: %(client_addr)s - "%(request_line)s" %(status_code)s',
                "use_colors": True,
            },
        },
        "handlers": get_handlers(),
    }

    log_config["loggers"] = {
        "": {"handlers": ["default"], "level": "INFO"},
        "uvicorn.error": {"handlers": ["default"], "level": "INFO"},
        "uvicorn.access": {
            "handlers": ["access"],
            "level": "INFO",
            "propagate": False,
        },
    }

    dictConfig(log_config)


logger = logging.getLogger("sortium-mesh-generation")
