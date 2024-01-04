import logging.config

log_config = {
    "version": 1,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "custom_format",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": "logs/logfile.log",
            "formatter": "custom_format",
        },
    },
    "formatters": {
        "custom_format": {
            "format": "%(asctime)s [%(levelname)s] %(process)d, %(funcName)s, %(lineno)d: %(message)s",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["file"],
    },
}

logging.config.dictConfig(log_config)


def get_log():
    return logging.getLogger("LOG")
