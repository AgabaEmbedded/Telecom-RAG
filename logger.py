import logging

logging.basicConfig(level= logging.DEBUG, filename = "log.log", filemode = "w",
                    format = "%(asctime)s: %(name)s: %(levelname)s: %(message)s")

logging.debug("debug")
logging.info("info")
logging.warning("warning")
logging.error("error")
logging.critical("critical")

print(__name__)