"""
End-to-end tests for Struckdown Playground using Playwright.

These tests verify the full user experience including:
- Page loading and rendering
- Editor interactions
- Real-time syntax analysis
- Input panel population
- Output rendering
- Pin functionality
- Batch mode
- localStorage persistence
- Keyboard shortcuts

Run with: uv run pytest struckdown/tests/test_playground_e2e.py -v
Requires: playwright browsers installed (playwright install)
"""

import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Skip all tests if playwright not installed
pytest.importorskip("playwright")

from playwright.sync_api import Page, expect


def set_editor_content(page: Page, content: str):
    """Set editor content, handling both CodeMirror and textarea."""
    page.wait_for_timeout(300)  # Wait for potential CodeMirror load

    cm_editor = page.locator(".cm-editor")
    if cm_editor.count() > 0:
        # CodeMirror - select all and replace
        cm_content = page.locator(".cm-content")
        cm_content.click()
        page.keyboard.press("Control+a")
        page.keyboard.press("Meta+a")  # For Mac
        page.keyboard.type(content)
    else:
        # Fallback textarea
        editor = page.locator("#editor-textarea")
        editor.fill(content)


@pytest.fixture(scope="module")
def playwright_browser():
    """Create a browser instance for all tests in this module."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def page(playwright_browser):
    """Create a new page for each test."""
    context = playwright_browser.new_context()
    page = context.new_page()
    yield page
    page.close()
    context.close()


@pytest.fixture
def temp_sd_file(tmp_path):
    """Create a temporary .sd file for testing."""
    sd_file = tmp_path / "test.sd"
    sd_file.write_text("Tell me about {{topic}}\n\n[[response]]")
    return sd_file


@pytest.fixture
def playground_server(temp_sd_file):
    """Start a playground server in a background thread."""
    from struckdown.playground import create_app, find_available_port

    port = find_available_port(start=19000)  # Use high port to avoid conflicts
    app = create_app(prompt_file=temp_sd_file, remote_mode=False)
    app.config["TESTING"] = True

    server_thread = threading.Thread(
        target=lambda: app.run(
            host="localhost", port=port, debug=False, use_reloader=False
        ),
        daemon=True,
    )
    server_thread.start()

    # Wait for server to start
    time.sleep(0.5)

    class ServerInfo:
        url = f"http://localhost:{port}"
        file = temp_sd_file

    yield ServerInfo()


@pytest.fixture
def playground_server_empty(tmp_path):
    """Start a playground server with an empty file."""
    from struckdown.playground import create_app, find_available_port

    sd_file = tmp_path / "empty.sd"
    sd_file.write_text("")

    port = find_available_port(start=19100)
    app = create_app(prompt_file=sd_file, remote_mode=False)
    app.config["TESTING"] = True

    server_thread = threading.Thread(
        target=lambda: app.run(
            host="localhost", port=port, debug=False, use_reloader=False
        ),
        daemon=True,
    )
    server_thread.start()
    time.sleep(0.5)

    class ServerInfo:
        url = f"http://localhost:{port}"
        file = sd_file

    yield ServerInfo()


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for batch testing."""
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("topic,style\nanimals,funny\nfood,serious\nmusic,casual\n")
    return csv_file


class TestPageLoading:
    """Tests for basic page loading and rendering."""

    def test_page_loads_successfully(self, page: Page, playground_server):
        """Page loads without errors."""
        page.goto(playground_server.url)
        expect(page).to_have_title("test.sd - Struckdown Playground")

    def test_editor_shows_file_content(self, page: Page, playground_server):
        """Editor contains the file content (CodeMirror or textarea)."""
        page.goto(playground_server.url)
        # Wait a bit for CodeMirror module to potentially load
        page.wait_for_timeout(500)

        # Check for CodeMirror first, then textarea fallback
        cm_editor = page.locator(".cm-editor")
        textarea = page.locator("#editor-textarea")

        if cm_editor.count() > 0:
            # CodeMirror loaded - check content in .cm-content
            content = page.locator(".cm-content").inner_text()
            assert "{{topic}}" in content
            assert "[[response]]" in content
        else:
            # Fallback textarea
            expect(textarea).to_have_value("Tell me about {{topic}}\n\n[[response]]")

    def test_filename_shown_in_header(self, page: Page, playground_server):
        """Filename is displayed in the header."""
        page.goto(playground_server.url)
        expect(page.locator("text=test.sd")).to_be_visible()

    def test_settings_panel_collapsed_by_default(self, page: Page, playground_server):
        """Settings panel is collapsed initially."""
        page.goto(playground_server.url)
        settings_panel = page.locator("#settings-panel")
        expect(settings_panel).not_to_be_visible()

    def test_inputs_panel_collapsed_by_default(self, page: Page, playground_server):
        """Inputs panel is collapsed initially."""
        page.goto(playground_server.url)
        inputs_panel = page.locator("#inputs-panel")
        expect(inputs_panel).not_to_be_visible()

    def test_bootstrap_icons_loaded(self, page: Page, playground_server):
        """Bootstrap icons CSS is loaded."""
        page.goto(playground_server.url)
        # Check that icon elements exist
        expect(page.locator(".bi-gear").first).to_be_visible()


class TestEditorInteractions:
    """Tests for editor text input and modifications."""

    def test_can_type_in_editor(self, page: Page, playground_server_empty):
        """User can type text into the editor."""
        page.goto(playground_server_empty.url)
        page.wait_for_timeout(500)

        # Try CodeMirror first
        cm_editor = page.locator(".cm-editor")
        if cm_editor.count() > 0:
            # CodeMirror - type into content area
            cm_content = page.locator(".cm-content")
            cm_content.click()
            page.keyboard.type("Hello [[world]]")
            content = cm_content.inner_text()
            assert "Hello [[world]]" in content
        else:
            # Fallback textarea
            editor = page.locator("#editor-textarea")
            editor.fill("Hello [[world]]")
            expect(editor).to_have_value("Hello [[world]]")

    def test_editor_preserves_content_on_focus_change(
        self, page: Page, playground_server
    ):
        """Editor content persists when focus moves elsewhere."""
        page.goto(playground_server.url)
        page.wait_for_timeout(500)

        # Get initial content
        cm_editor = page.locator(".cm-editor")
        if cm_editor.count() > 0:
            original_content = page.locator(".cm-content").inner_text()
            page.locator("body").click()
            page.wait_for_timeout(100)
            new_content = page.locator(".cm-content").inner_text()
            assert original_content == new_content
        else:
            editor = page.locator("#editor-textarea")
            original_value = editor.input_value()
            page.locator("body").click()
            expect(editor).to_have_value(original_value)


class TestSyntaxAnalysis:
    """Tests for real-time syntax analysis and error display."""

    def test_analysis_triggers_on_input(self, page: Page, playground_server_empty):
        """Typing triggers syntax analysis after debounce."""
        page.goto(playground_server_empty.url)

        # Type valid syntax with a variable
        set_editor_content(page, "{{name}} says [[greeting]]")

        # Wait for debounced analysis (500ms + network time)
        page.wait_for_timeout(800)

        # Inputs panel should be updated
        page.locator("button:has-text('Inputs')").click()
        expect(page.locator(".input-field[name='name']")).to_be_visible()

    def test_parse_error_shows_in_banner(self, page: Page, playground_server_empty):
        """Invalid syntax shows error in banner."""
        page.goto(playground_server_empty.url)

        # Type invalid syntax (unclosed slot)
        set_editor_content(page, "[[broken")

        # Wait for analysis
        page.wait_for_timeout(800)

        # Error banner should be visible
        error_banner = page.locator("#error-banner")
        expect(error_banner).to_be_visible()

    def test_error_banner_hides_on_valid_syntax(
        self, page: Page, playground_server_empty
    ):
        """Error banner disappears when syntax becomes valid."""
        page.goto(playground_server_empty.url)

        # Type invalid syntax
        set_editor_content(page, "[[broken")
        page.wait_for_timeout(800)
        expect(page.locator("#error-banner")).to_be_visible()

        # Fix the syntax
        set_editor_content(page, "[[fixed]]")
        page.wait_for_timeout(800)

        # Error should be hidden
        expect(page.locator("#error-banner")).not_to_be_visible()


class TestInputsPanel:
    """Tests for the dynamic inputs panel."""

    def test_inputs_panel_opens_on_click(self, page: Page, playground_server):
        """Clicking Inputs button opens the panel."""
        page.goto(playground_server.url)
        page.locator("button:has-text('Inputs')").click()
        expect(page.locator("#inputs-panel")).to_be_visible()

    def test_input_field_created_for_variable(self, page: Page, playground_server):
        """Input field appears for {{variable}} in syntax."""
        page.goto(playground_server.url)

        # Wait for initial analysis
        page.wait_for_timeout(800)

        # Open inputs panel
        page.locator("button:has-text('Inputs')").click()

        # Should have a field for 'topic'
        expect(page.locator(".input-field[name='topic']")).to_be_visible()

    def test_input_field_removed_when_variable_removed(
        self, page: Page, playground_server
    ):
        """Input field disappears when variable is removed from syntax."""
        page.goto(playground_server.url)
        page.wait_for_timeout(800)

        # Open inputs panel
        page.locator("button:has-text('Inputs')").click()
        expect(page.locator(".input-field[name='topic']")).to_be_visible()

        # Remove the variable from syntax
        set_editor_content(page, "Just a response [[response]]")
        page.wait_for_timeout(800)

        # Field should be gone
        expect(page.locator(".input-field[name='topic']")).not_to_be_visible()

    def test_multiple_input_fields(self, page: Page, playground_server_empty):
        """Multiple variables create multiple input fields."""
        page.goto(playground_server_empty.url)

        set_editor_content(page, "{{name}} from {{city}} says [[greeting]]")
        page.wait_for_timeout(800)

        page.locator("button:has-text('Inputs')").click()

        expect(page.locator(".input-field[name='name']")).to_be_visible()
        expect(page.locator(".input-field[name='city']")).to_be_visible()


class TestSettingsPanel:
    """Tests for the settings panel."""

    def test_settings_panel_opens_on_click(self, page: Page, playground_server):
        """Clicking Settings button opens the panel."""
        page.goto(playground_server.url)
        page.locator("button:has-text('Settings')").click()
        expect(page.locator("#settings-panel")).to_be_visible()

    def test_model_input_exists(self, page: Page, playground_server):
        """Model input field is present in settings."""
        page.goto(playground_server.url)
        page.locator("button:has-text('Settings')").click()
        expect(page.locator("#model-input")).to_be_visible()


class TestModeToggle:
    """Tests for the Single/Batch mode toggle in the Inputs panel."""

    def test_mode_toggle_exists_in_inputs_panel(self, page: Page, playground_server):
        """Single/Batch mode toggle is in the Inputs panel."""
        page.goto(playground_server.url)
        page.locator("button:has-text('Inputs')").click()
        expect(page.locator("#mode-single")).to_be_visible()
        expect(page.locator("#mode-batch")).to_be_visible()

    def test_single_inputs_shown_by_default(self, page: Page, playground_server):
        """Single input fields shown by default."""
        page.goto(playground_server.url)
        page.locator("button:has-text('Inputs')").click()

        expect(page.locator("#single-inputs-section")).to_be_visible()
        expect(page.locator("#batch-upload-section")).not_to_be_visible()

    def test_batch_upload_shown_when_batch_selected(
        self, page: Page, playground_server
    ):
        """Batch file upload appears when batch mode selected."""
        page.goto(playground_server.url)
        page.locator("button:has-text('Inputs')").click()

        # Initially single mode
        expect(page.locator("#single-inputs-section")).to_be_visible()
        expect(page.locator("#batch-upload-section")).not_to_be_visible()

        # Select batch mode
        page.locator("label[for='mode-batch']").click()

        # Now batch section visible, single hidden
        expect(page.locator("#single-inputs-section")).not_to_be_visible()
        expect(page.locator("#batch-upload-section")).to_be_visible()

    def test_switching_back_to_single_mode(self, page: Page, playground_server):
        """Can switch back to single mode after selecting batch."""
        page.goto(playground_server.url)
        page.locator("button:has-text('Inputs')").click()

        # Switch to batch
        page.locator("label[for='mode-batch']").click()
        expect(page.locator("#batch-upload-section")).to_be_visible()

        # Switch back to single
        page.locator("label[for='mode-single']").click()
        expect(page.locator("#single-inputs-section")).to_be_visible()
        expect(page.locator("#batch-upload-section")).not_to_be_visible()


class TestLocalStoragePersistence:
    """Tests for localStorage persistence of inputs."""

    def test_session_id_in_url_hash(self, page: Page, playground_server):
        """Session ID is added to URL hash on load."""
        page.goto(playground_server.url)
        page.wait_for_timeout(100)

        url = page.url
        assert "#s=" in url

    def test_input_values_persist_on_reload(self, page: Page, playground_server):
        """Input values are saved and restored after page reload."""
        page.goto(playground_server.url)
        page.wait_for_timeout(800)

        # Open inputs panel and fill a value
        page.locator("button:has-text('Inputs')").click()
        input_field = page.locator(".input-field[name='topic']")
        input_field.fill("programming")

        # Wait for localStorage save (debounced)
        page.wait_for_timeout(600)

        # Get the current URL (with session ID)
        url_with_session = page.url

        # Reload the page
        page.goto(url_with_session)
        page.wait_for_timeout(800)

        # Open inputs panel and check value
        page.locator("button:has-text('Inputs')").click()
        expect(page.locator(".input-field[name='topic']")).to_have_value("programming")

    def test_model_value_persists_on_reload(self, page: Page, playground_server):
        """Model selection is saved and restored after reload."""
        page.goto(playground_server.url)
        page.wait_for_timeout(500)

        # Get URL with session
        url_with_session = page.url

        # Open settings and set model
        page.locator("button:has-text('Settings')").click()
        model_input = page.locator("#model-input")
        model_input.fill("openai/gpt-4o")

        # Wait for save
        page.wait_for_timeout(600)

        # Reload
        page.goto(url_with_session)
        page.wait_for_timeout(800)

        # Check value restored
        page.locator("button:has-text('Settings')").click()
        expect(page.locator("#model-input")).to_have_value("openai/gpt-4o")

    def test_different_sessions_dont_share_data(self, page: Page, playground_server):
        """Different session IDs have separate localStorage."""
        # First session
        page.goto(playground_server.url)
        page.wait_for_timeout(800)

        page.locator("button:has-text('Inputs')").click()
        page.locator(".input-field[name='topic']").fill("session1_value")
        page.wait_for_timeout(600)

        # Navigate to a new session (no hash)
        page.goto(playground_server.url.split("#")[0])
        page.wait_for_timeout(800)

        # Should have empty input (different session)
        page.locator("button:has-text('Inputs')").click()
        input_val = page.locator(".input-field[name='topic']").input_value()
        assert input_val != "session1_value"


class TestKeyboardShortcuts:
    """Tests for keyboard shortcuts."""

    def test_ctrl_s_triggers_save_and_run(self, page: Page, playground_server):
        """Ctrl+S triggers Save & Run action."""
        page.goto(playground_server.url)
        page.wait_for_timeout(500)

        # Focus the editor (CodeMirror or textarea)
        cm_editor = page.locator(".cm-editor")
        if cm_editor.count() > 0:
            page.locator(".cm-content").click()
        else:
            page.locator("#editor-textarea").focus()

        # Press Ctrl+S
        page.keyboard.press("Control+s")

        # Button should show running state
        expect(page.locator("#save-run-btn")).to_contain_text("Running")

    def test_cmd_s_triggers_save_and_run_mac(self, page: Page, playground_server):
        """Cmd+S (Meta+S) triggers Save & Run action."""
        page.goto(playground_server.url)
        page.wait_for_timeout(500)

        # Focus the editor (CodeMirror or textarea)
        cm_editor = page.locator(".cm-editor")
        if cm_editor.count() > 0:
            page.locator(".cm-content").click()
        else:
            page.locator("#editor-textarea").focus()

        # Press Cmd+S (Meta+S)
        page.keyboard.press("Meta+s")

        # Button should show running state
        expect(page.locator("#save-run-btn")).to_contain_text("Running")


class TestSaveAndRun:
    """Tests for the Save & Run functionality."""

    def test_save_and_run_button_shows_spinner(self, page: Page, playground_server):
        """Save & Run button shows spinner while running."""
        page.goto(playground_server.url)
        page.wait_for_timeout(500)

        # Click Save & Run
        page.locator("#save-run-btn").click()

        # Should show spinner
        expect(page.locator("#save-run-btn")).to_contain_text("Running")

    def test_save_updates_file(self, page: Page, playground_server):
        """Saving writes content to the file."""
        page.goto(playground_server.url)
        page.wait_for_timeout(500)

        # Modify content
        set_editor_content(page, "New content [[slot]]")

        # Click Save & Run (which saves first)
        page.locator("#save-run-btn").click()
        page.wait_for_timeout(1000)

        # Verify file was updated
        assert playground_server.file.read_text() == "New content [[slot]]"


class TestOutputDisplay:
    """Tests for output rendering (requires mocking LLM)."""

    def test_outputs_container_exists(self, page: Page, playground_server):
        """Outputs container is present on page."""
        page.goto(playground_server.url)
        expect(page.locator("#outputs-container")).to_be_visible()

    def test_initial_outputs_show_placeholder(self, page: Page, playground_server):
        """Initially outputs show placeholder text."""
        page.goto(playground_server.url)
        expect(page.locator("#outputs-container")).to_contain_text("Save & Run")


class TestBatchMode:
    """Tests for batch mode functionality."""

    def test_batch_file_upload_shows_info(
        self, page: Page, playground_server, sample_csv
    ):
        """Uploading a CSV shows file info."""
        page.goto(playground_server.url)
        page.wait_for_timeout(500)

        # Open inputs panel and select batch mode
        page.locator("button:has-text('Inputs')").click()
        page.locator("label[for='mode-batch']").click()

        # Upload file
        page.locator("#batch-file").set_input_files(str(sample_csv))

        # Wait for upload
        page.wait_for_timeout(1000)

        # Should show file info
        expect(page.locator("#batch-file-info")).to_be_visible()
        expect(page.locator("#batch-filename")).to_contain_text("sample.csv")
        expect(page.locator("#batch-row-count")).to_contain_text("3")

    def test_clear_batch_file(self, page: Page, playground_server, sample_csv):
        """Can clear uploaded batch file."""
        page.goto(playground_server.url)
        page.wait_for_timeout(500)

        # Open inputs panel and upload file
        page.locator("button:has-text('Inputs')").click()
        page.locator("label[for='mode-batch']").click()
        page.locator("#batch-file").set_input_files(str(sample_csv))
        page.wait_for_timeout(1000)

        # Clear file
        page.locator("text=Remove").click()

        # Info should be hidden
        expect(page.locator("#batch-file-info")).not_to_be_visible()


class TestOutputOrdering:
    """Tests for output ordering matching template order."""

    def test_slots_defined_order(self, page: Page, playground_server_empty):
        """Analysis returns slots in template order."""
        page.goto(playground_server_empty.url)

        # Enter template with slots in specific order
        set_editor_content(page, "[[first]]\n[[second]]\n[[third]]")
        page.wait_for_timeout(800)

        # Make API call to check order
        response = page.request.post(
            f"{playground_server_empty.url}/api/analyse",
            data=json.dumps({"syntax": "[[first]]\n[[second]]\n[[third]]"}),
            headers={"Content-Type": "application/json"},
        )

        data = response.json()
        assert data["slots_defined"] == ["first", "second", "third"]


class TestStatusBar:
    """Tests for the status bar."""

    def test_status_bar_shows_ready(self, page: Page, playground_server):
        """Status bar shows 'Ready' initially."""
        page.goto(playground_server.url)
        expect(page.locator("#status-text")).to_contain_text("Ready")


class TestResponsiveLayout:
    """Tests for responsive layout behavior."""

    def test_two_column_layout(self, page: Page, playground_server):
        """Page has two-column layout on desktop."""
        page.goto(playground_server.url)

        editor_col = page.locator("#editor-column")
        outputs_col = page.locator("#outputs-column")

        expect(editor_col).to_be_visible()
        expect(outputs_col).to_be_visible()

        # Both should be side by side (check they have similar y position)
        editor_box = editor_col.bounding_box()
        outputs_box = outputs_col.bounding_box()

        # Should be roughly same y position (within 50px tolerance)
        assert abs(editor_box["y"] - outputs_box["y"]) < 50
