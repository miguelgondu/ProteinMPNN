import logging

# Prevent "No handler found" warnings for library users who don't configure logging
logging.getLogger("proteinmpnn").addHandler(logging.NullHandler())


def main() -> None:
    print("Hello from proteinmpnn!")
