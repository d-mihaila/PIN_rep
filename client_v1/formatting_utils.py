# %%
import textwrap
from benedict import benedict
from langchain_core.documents import Document


def _fixed_width_wrap(text, width: int = 70, join_str: str = "\n"):
    return join_str.join(textwrap.wrap(text, width=width))


def fixed_width_wrap(text, width: int = 70, join_str: str = "\n", split_str="\n"):
    return join_str.join(
        [
            _fixed_width_wrap(t, width=width, join_str=join_str)
            for t in text.split(split_str)
        ]
    )


def format_doc_minimal(d, fixed_width=False):
    if isinstance(d, Document):
        _cont = d.page_content
        _meta = benedict(d.metadata)
    else:
        _cont = d["page_content"]
        _meta = benedict(d["metadata"])

    if fixed_width:
        _cont = _fixed_width_wrap(_cont)

    return """\
Title:\t{title}
Published on:\t{pubdate}
Source:\t{source_name} ({source_country})
Chunk Content:

\t{cont}
""".format(
        d=d,
        title=_meta.get("title"),
        pubdate=_meta.get("pubdate"),
        source_name=_meta.get("source.host") or _meta.get("source.id"),
        source_country=_meta.get("source.country", "n/a"),
        cont=_cont,
    )


def format_docs(docs, doc_fn=format_doc_minimal, **kwargs):
    return "\n---\n".join([doc_fn(d, **kwargs) for d in docs])
