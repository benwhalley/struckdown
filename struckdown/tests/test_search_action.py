"""
Tests for the @search action using ddgs package.
"""

from unittest.mock import MagicMock, patch

import pytest

from struckdown.actions.search import (DEFAULT_MAX_TOKENS, SearchResult,
                                       fetch_page_content, search_action,
                                       search_and_fetch)


class TestSearchResult:
    """Test SearchResult dataclass."""

    def test_str_without_content(self):
        """SearchResult formats correctly without content."""
        result = SearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="This is a snippet",
        )
        formatted = str(result)
        assert "## Test Title" in formatted
        assert "https://example.com" in formatted
        assert "This is a snippet" in formatted
        assert "Content" not in formatted

    def test_str_with_content(self):
        """SearchResult formats correctly with content."""
        result = SearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="This is a snippet",
            content="Full page content here",
        )
        formatted = str(result)
        assert "### Content:" in formatted
        assert "Full page content here" in formatted


class TestSearchAndFetch:
    """Test search_and_fetch function."""

    @patch("struckdown.actions.search.fetch_page_content")
    @patch("struckdown.actions.search.DDGS")
    def test_search_fetches_by_default(self, mock_ddgs_class, mock_fetch):
        """Search fetches page content by default."""
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = [
            {"title": "Result 1", "href": "https://example.com/1", "body": "Snippet 1"},
            {"title": "Result 2", "href": "https://example.com/2", "body": "Snippet 2"},
        ]
        mock_ddgs_class.return_value = mock_ddgs
        mock_fetch.return_value = "Fetched content"

        results = search_and_fetch("test query", max_results=2)

        assert len(results) == 2
        assert results[0].title == "Result 1"
        assert results[0].url == "https://example.com/1"
        assert results[0].snippet == "Snippet 1"
        assert results[0].content == "Fetched content"
        assert mock_fetch.call_count == 2

    @patch("struckdown.actions.search.fetch_page_content")
    @patch("struckdown.actions.search.DDGS")
    def test_search_with_max_tokens(self, mock_ddgs_class, mock_fetch):
        """Search passes max_tokens to fetch."""
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = [
            {"title": "Result 1", "href": "https://example.com/1", "body": "Snippet 1"},
        ]
        mock_ddgs_class.return_value = mock_ddgs
        mock_fetch.return_value = "Fetched content"

        results = search_and_fetch("test query", max_results=1, max_tokens=5000)

        assert len(results) == 1
        # Check that fetch was called with correct max_tokens
        call_args = mock_fetch.call_args
        assert call_args[0][2] == 5000  # max_tokens arg


class TestSearchAction:
    """Test the @search action registration and execution."""

    @patch("struckdown.actions.search.search_and_fetch")
    def test_search_action_embeds_by_default(self, mock_search):
        """Search action embeds full content by default."""
        mock_search.return_value = [
            SearchResult(
                title="Apple - Wikipedia",
                url="https://en.wikipedia.org/wiki/Apple",
                snippet="An apple is a round fruit...",
                content="# Apple\n\nApples are delicious fruits...",
            ),
        ]

        result = search_action(
            context={},
            query="wikipedia apples",
            max_results=1,
        )

        assert "# Search Results for: wikipedia apples" in result
        assert "Apple - Wikipedia" in result
        assert "https://en.wikipedia.org/wiki/Apple" in result
        assert "Apples are delicious fruits" in result

    @patch("struckdown.actions.search.search_and_fetch")
    def test_search_action_embed_false(self, mock_search):
        """Search action with embed=False returns summary only."""
        mock_search.return_value = [
            SearchResult(
                title="Apple - Wikipedia",
                url="https://en.wikipedia.org/wiki/Apple",
                snippet="An apple is a round fruit...",
                content="# Apple\n\nApples are delicious fruits...",
            ),
        ]

        result = search_action(
            context={},
            query="wikipedia apples",
            max_results=1,
            embed=False,
        )

        assert "Apple - Wikipedia" in result
        assert "https://en.wikipedia.org/wiki/Apple" in result
        assert "An apple is a round fruit" in result
        # Full content should NOT be embedded
        assert "Apples are delicious fruits" not in result

    @patch("struckdown.actions.search.search_and_fetch")
    def test_search_action_no_results(self, mock_search):
        """Search action handles no results gracefully."""
        mock_search.return_value = []

        result = search_action(context={}, query="xyznonexistent123")

        assert "No results found" in result


class TestFetchPageContent:
    """Test page content fetching."""

    @patch("struckdown.actions.search.html_to_markdown")
    @patch("struckdown.actions.search.extract_readable")
    @patch("struckdown.actions.search.fetch_url")
    def test_fetch_page_content_success(self, mock_fetch, mock_readable, mock_md):
        """Successfully fetches and converts page content."""
        mock_fetch.return_value = ("<html><body>Content</body></html>", "text/html")
        mock_readable.return_value = "<p>Readable content</p>"
        mock_md.return_value = "Markdown content"

        result = fetch_page_content("https://example.com")

        assert result == "Markdown content"
        mock_fetch.assert_called_once()

    @patch("struckdown.actions.search.fetch_url")
    def test_fetch_page_content_failure(self, mock_fetch):
        """Returns empty string on fetch failure."""
        mock_fetch.side_effect = Exception("Network error")

        result = fetch_page_content("https://example.com")

        assert result == ""


# =============================================================================
# Integration test - requires network (marked for optional running)
# =============================================================================


@pytest.mark.integration
@pytest.mark.skip(reason="Requires network access - run manually with -m integration")
class TestSearchIntegration:
    """Integration tests that hit real network."""

    def test_wikipedia_apples_search(self):
        """Search for wikipedia apples and verify results."""
        results = search_and_fetch(
            query="wikipedia apples",
            max_results=2,
            fetch_content=True,
            max_chars_per_page=4000,
        )

        assert len(results) >= 1
        # should find wikipedia
        urls = [r.url for r in results]
        assert any("wikipedia" in url.lower() for url in urls)

        # should have fetched content
        assert any(r.content for r in results)

        # print for manual inspection
        for r in results:
            print(f"\n=== {r.title} ===")
            print(f"URL: {r.url}")
            print(f"Content length: {len(r.content)} chars")
            print(f"Content preview: {r.content[:500]}...")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
