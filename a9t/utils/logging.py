import logging.config

log_config = {
    "version": 1,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "custom_format",
        },
    },
    "formatters": {
        "custom_format": {
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
        },
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console"],
    },
}

logging.config.dictConfig(log_config)


def get_log():
    return logging.getLogger("LOG")

