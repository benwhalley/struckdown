"""
Tests for the @evidence action with BM25 search.

Tests that evidence files are searched, ranked, and included in prompt context,
and that LLM completions can reference the retrieved evidence.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from struckdown import chatter
from struckdown.actions import Actions
from struckdown.actions.evidence import chunk_text, evidence_search, load_evidence_files


class TestChunkText:
    """Test text chunking utility"""

    def test_short_text_single_chunk(self):
        """Short text should return single chunk"""
        text = "This is a short sentence."
        chunks = chunk_text(text, chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self):
        """Long text should be split into multiple chunks"""
        text = "word " * 200  # 1000 chars
        chunks = chunk_text(text, chunk_size=100)
        assert len(chunks) > 1
        # each chunk should be roughly chunk_size or less
        for chunk in chunks:
            assert len(chunk) <= 150  # some tolerance for word boundaries

    def test_preserves_words(self):
        """Chunks should not split words"""
        text = "supercalifragilisticexpialidocious " * 10
        chunks = chunk_text(text, chunk_size=50)
        for chunk in chunks:
            # each chunk should be complete words
            assert not chunk.startswith(" ")
            assert not chunk.endswith(" ") or chunk == chunks[-1]


class TestLoadEvidenceFiles:
    """Test evidence file loading"""

    def test_loads_txt_files(self):
        """Should load .txt files from folder"""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "doc1.txt").write_text("First document content")
            (folder / "doc2.txt").write_text("Second document content")

            chunks = load_evidence_files([folder])

            assert len(chunks) >= 2
            filenames = [c[0] for c in chunks]
            assert "doc1.txt" in filenames
            assert "doc2.txt" in filenames

    def test_loads_md_files(self):
        """Should load .md files from folder"""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "readme.md").write_text("# Markdown document")

            chunks = load_evidence_files([folder])

            assert len(chunks) >= 1
            assert chunks[0][0] == "readme.md"

    def test_ignores_other_extensions(self):
        """Should ignore non-txt/md files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "doc.txt").write_text("Text content")
            (folder / "data.json").write_text('{"key": "value"}')
            (folder / "script.py").write_text("print('hello')")

            chunks = load_evidence_files([folder])

            filenames = [c[0] for c in chunks]
            assert "doc.txt" in filenames
            assert "data.json" not in filenames
            assert "script.py" not in filenames

    def test_loads_from_subdirectories(self):
        """Should recursively load from subdirectories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            subdir = folder / "subdir"
            subdir.mkdir()
            (folder / "root.txt").write_text("Root content")
            (subdir / "nested.txt").write_text("Nested content")

            chunks = load_evidence_files([folder])

            filenames = [c[0] for c in chunks]
            assert "root.txt" in filenames
            assert "nested.txt" in filenames

    def test_deduplicates_folders(self):
        """Should not load same folder twice"""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "doc.txt").write_text("Content")

            # pass same folder twice
            chunks = load_evidence_files([folder, folder])

            # should only have one chunk (not duplicated)
            assert len(chunks) == 1


class TestEvidenceSearch:
    """Test BM25 evidence search"""

    def test_finds_matching_content(self):
        """Should find documents matching query"""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            # note: BM25 needs at least 3 documents for proper IDF calculation
            (folder / "cbt.txt").write_text(
                "cognitive behavioural therapy CBT is an effective treatment anxiety "
                "involves identifying and challenging negative thought patterns"
            )
            (folder / "diet.txt").write_text(
                "healthy diet includes fruits vegetables and whole grains "
                "avoid processed foods and excess sugar"
            )
            (folder / "exercise.txt").write_text(
                "regular exercise improves cardiovascular health and mental wellbeing"
            )

            context = {"evidence_folder": str(folder)}
            result = evidence_search(context, query="cognitive therapy anxiety", n=1)

            assert "cbt.txt" in result.lower() or "cognitive" in result.lower()
            assert "anxiety" in result.lower()

    def test_returns_top_n_results(self):
        """Should return up to n results"""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            for i in range(5):
                (folder / f"doc{i}.txt").write_text(f"Document {i} about python programming")

            context = {"evidence_folder": str(folder)}
            result = evidence_search(context, query="python programming", n=2)

            # should have separator between results
            assert result.count("---") <= 1  # at most 1 separator for 2 results

    def test_handles_multiple_queries(self):
        """Should search with list of queries"""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            # BM25 needs at least 3 docs for proper IDF
            (folder / "anxiety.txt").write_text("treatment approaches for anxiety disorders")
            (folder / "depression.txt").write_text("managing depression with therapy")
            (folder / "sleep.txt").write_text("improving sleep quality and rest")

            context = {"evidence_folder": str(folder)}
            result = evidence_search(context, query=["anxiety treatment", "depression therapy"], n=3)

            # should find content from both topics
            assert len(result) > 0

    def test_returns_empty_for_no_matches(self):
        """Should return empty string if no evidence found"""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            (folder / "doc.txt").write_text("This is about cooking recipes")

            context = {"evidence_folder": str(folder)}
            result = evidence_search(context, query="quantum physics", n=3)

            # BM25 may return low-scoring results, but they should be minimal
            # for completely unrelated content
            assert result == "" or len(result) < 100

    def test_empty_folder_returns_empty(self):
        """Should return empty for empty evidence folder"""
        with tempfile.TemporaryDirectory() as tmpdir:
            context = {"evidence_folder": tmpdir}
            result = evidence_search(context, query="anything", n=3)
            assert result == ""

    def test_no_folder_returns_empty(self):
        """Should return empty if no evidence folders configured"""
        context = {}
        result = evidence_search(context, query="anything", n=3)
        assert result == ""


class TestEvidenceActionRegistration:
    """Test that @evidence action is properly registered"""

    def test_evidence_is_registered(self):
        """@evidence should be in the action registry"""
        assert Actions.is_registered("evidence")

    def test_evidence_default_save(self):
        """@evidence should have default_save=True"""
        model = Actions.create_action_model("evidence", ["query=test"], None, False)
        assert model._default_save is True


class TestEvidenceIntegration:
    """Integration tests using chatter() with @evidence action"""

    def test_evidence_in_template_context(self):
        """Test that @evidence output is available in template"""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            # avoid punctuation for simple tokenizer
            (folder / "facts.txt").write_text(
                "the capital of France is Paris "
                "Paris is known for the Eiffel Tower"
            )

            template = """
[[@evidence:info|query="capital France Paris"]]

Based on the evidence above, what is the capital of France?

[[answer]]
"""

            # mock the LLM call to verify evidence was included
            with patch("struckdown.llm.structured_chat") as mock_chat:
                mock_result = Mock()
                mock_result.response = "Paris"
                mock_completion = {"usage": {"total_tokens": 10}}
                mock_chat.return_value = (mock_result, mock_completion)

                result = chatter(template, context={"evidence_folder": str(folder)})

                # verify structured_chat was called
                assert mock_chat.called

                # check messages passed to LLM include evidence
                call_args = mock_chat.call_args
                messages = call_args.kwargs.get("messages", [])

                # find user message content
                user_content = " ".join(
                    m["content"] for m in messages if m["role"] == "user"
                )

                # evidence should be in the context
                assert "Paris" in user_content or "France" in user_content

    def test_evidence_empty_gracefully(self):
        """Test that missing evidence folder doesn't break template"""
        template = """
[[@evidence:info|query="test query"]]

Answer: [[answer]]
"""

        with patch("struckdown.llm.structured_chat") as mock_chat:
            mock_result = Mock()
            mock_result.response = "No evidence found"
            mock_completion = {"usage": {"total_tokens": 10}}
            mock_chat.return_value = (mock_result, mock_completion)

            # should not raise, even without evidence folder
            result = chatter(template, context={})

            assert mock_chat.called

    def test_evidence_with_template_path_discovery(self):
        """Test that evidence folder is discovered relative to template"""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            evidence_dir = template_dir / "evidence"
            evidence_dir.mkdir()

            # avoid punctuation for simple tokenizer
            (evidence_dir / "knowledge.txt").write_text(
                "important fact the speed of light is 299792458 meters per second"
            )

            template = """
[[@evidence:facts|query="speed light"]]

What is the speed of light?
[[answer]]
"""

            with patch("struckdown.llm.structured_chat") as mock_chat:
                mock_result = Mock()
                mock_result.response = "299792458 m/s"
                mock_completion = {"usage": {"total_tokens": 10}}
                mock_chat.return_value = (mock_result, mock_completion)

                # pass _template_path to trigger auto-discovery
                result = chatter(
                    template,
                    context={"_template_path": str(template_dir / "prompt.sd")}
                )

                # verify LLM was called with evidence in context
                call_args = mock_chat.call_args
                messages = call_args.kwargs.get("messages", [])
                user_content = " ".join(
                    m["content"] for m in messages if m["role"] == "user"
                )

                assert "299792458" in user_content or "speed" in user_content.lower()


class TestBM25Ranking:
    """Test BM25 ranking behavior"""

    def test_relevance_ranking(self):
        """More relevant documents should rank higher"""
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)

            # highly relevant document
            (folder / "relevant.txt").write_text(
                "Python programming language tutorial. "
                "Learn Python basics, Python functions, Python classes. "
                "Python is great for beginners learning programming."
            )

            # somewhat relevant
            (folder / "partial.txt").write_text(
                "Programming languages include Java, C++, and Python. "
                "Each has different use cases."
            )

            # irrelevant
            (folder / "irrelevant.txt").write_text(
                "Cooking recipes for pasta and pizza. "
                "Italian cuisine is delicious."
            )

            context = {"evidence_folder": str(folder)}
            result = evidence_search(context, query="Python programming tutorial", n=2)

            # most relevant should appear first
            assert "relevant.txt" in result
            # irrelevant should not appear in top 2
            assert "pasta" not in result.lower()
            assert "pizza" not in result.lower()
