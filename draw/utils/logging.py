import logging.config

log_config = {
    "version": 1,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "custom_format",
        },
        "file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "filename": "logs/logfile.log",
            "when": "midnight",  # rotate daily
            "interval": 1,  # every day
            "backupCount": 7,  # keep 7 backup copies
            "encoding": "utf-8",
            "level": "INFO",
            "formatter": "custom_format",
        },
    },
    "formatters": {
        "custom_format": {
            "format": "%(asctime)s, PID: %(process)5d [%(levelname)7s] %(module)s.%(funcName)s@%(lineno)d: %(message)s",
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
