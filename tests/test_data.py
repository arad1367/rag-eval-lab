import pytest

from rag_lab.data import load_jsonl, make_chunks, save_jsonl, split_words
from scripts.fetch_papers import parse_feed


def test_split_words_with_overlap():
    text = " ".join(str(i) for i in range(10))  # "0 1 2 ... 9"
    pieces = split_words(text, chunk_size=4, overlap=1)
    assert pieces == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]


def test_split_words_short_text_is_one_chunk():
    assert split_words("just three words", chunk_size=50, overlap=10) == ["just three words"]


def test_split_words_rejects_bad_overlap():
    with pytest.raises(ValueError):
        split_words("a b c", chunk_size=3, overlap=3)


def test_make_chunks_adds_title_and_ids(papers):
    chunks = make_chunks(papers, chunk_size=5, overlap=1)
    first = chunks[0]
    assert first.paper_id == "0001"
    assert first.chunk_id == "0001-0"
    assert first.text.startswith(papers[0]["title"])
    assert len({c.paper_id for c in chunks}) == len(papers)


def test_jsonl_roundtrip(tmp_path, papers):
    path = tmp_path / "papers.jsonl"
    save_jsonl(papers, path)
    assert load_jsonl(path) == papers


ATOM_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2511.01234v2</id>
    <published>2025-11-03T18:00:00Z</published>
    <title>A  Test
      Title</title>
    <summary>  An abstract
      on two lines. </summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <arxiv:primary_category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>"""


def test_parse_arxiv_feed():
    [paper] = parse_feed(ATOM_SAMPLE)
    assert paper == {
        "id": "2511.01234",
        "title": "A Test Title",
        "abstract": "An abstract on two lines.",
        "authors": ["Ada Lovelace", "Alan Turing"],
        "published": "2025-11-03",
        "category": "cs.CL",
    }
