import io
import os
from fastapi import FastAPI
from .routes import main_router


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("VERSION")
    """
    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


description = """QA Bot to provide answers based on a given context. 🚀"""

app = FastAPI(
    title="QA Bot",
    description=description,
    version=read("VERSION")
)

app.include_router(main_router)
