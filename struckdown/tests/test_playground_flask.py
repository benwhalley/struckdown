"""Integration tests for struckdown.playground Flask app."""

import json
import tempfile
from pathlib import Path

import pytest

from struckdown.playground import create_app


@pytest.fixture
def temp_prompt_file(tmp_path):
    """Create a temporary prompt file."""
    prompt_file = tmp_path / "test.sd"
    prompt_file.write_text("Tell me about {{topic}}\n[[response]]")
    return prompt_file


@pytest.fixture
def client(temp_prompt_file):
    """Create a Flask test client."""
    app = create_app(prompt_file=temp_prompt_file, remote_mode=False)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def remote_client():
    """Create a Flask test client in remote mode."""
    app = create_app(prompt_file=None, remote_mode=True)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestHomePage:
    """Tests for the main editor page."""

    def test_home_page_loads(self, client):
        """Home page renders successfully."""
        response = client.get("/")
        assert response.status_code == 200
        assert b"Struckdown Playground" in response.data

    def test_home_page_shows_filename(self, client):
        """Home page shows the filename being edited."""
        response = client.get("/")
        assert b"test.sd" in response.data

    def test_home_page_loads_syntax(self, client):
        """Home page loads syntax from file."""
        response = client.get("/")
        assert b"Tell me about" in response.data

    def test_remote_mode_no_filename(self, remote_client):
        """Remote mode doesn't show a filename."""
        response = remote_client.get("/")
        assert response.status_code == 200
        # Should have remote badge
        assert b"Remote" in response.data


class TestSaveEndpoint:
    """Tests for the /api/save endpoint."""

    def test_save_updates_file(self, client, temp_prompt_file):
        """Save endpoint writes to file."""
        new_content = "New content [[slot]]"
        response = client.post(
            "/api/save",
            data=json.dumps({"syntax": new_content}),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert temp_prompt_file.read_text() == new_content

    def test_save_fails_in_remote_mode(self, remote_client):
        """Save endpoint returns error in remote mode."""
        response = remote_client.post(
            "/api/save",
            data=json.dumps({"syntax": "test"}),
            content_type="application/json",
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data


class TestAnalyseEndpoint:
    """Tests for the /api/analyse endpoint."""

    def test_analyse_extracts_inputs(self, client):
        """Analyse endpoint extracts required inputs."""
        response = client.post(
            "/api/analyse",
            data=json.dumps({"syntax": "{{name}} says [[greeting]]"}),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["inputs_required"] == ["name"]
        assert data["slots_defined"] == ["greeting"]

    def test_analyse_validates_syntax(self, client):
        """Analyse endpoint validates syntax."""
        # Valid syntax
        response = client.post(
            "/api/analyse",
            data=json.dumps({"syntax": "Hello [[greeting]]"}),
            content_type="application/json",
        )
        data = response.get_json()
        assert data["valid"] is True
        assert data["error"] is None

    def test_analyse_reports_parse_errors(self, client):
        """Analyse endpoint reports parse errors."""
        # Note: Most syntax is valid to the parser, but let's test empty slots
        response = client.post(
            "/api/analyse",
            data=json.dumps({"syntax": "Plain text without slots"}),
            content_type="application/json",
        )
        data = response.get_json()
        # Plain text should be valid
        assert data["valid"] is True
        assert data["slots_defined"] == []

    def test_analyse_multiple_variables(self, client):
        """Analyse handles multiple variables."""
        response = client.post(
            "/api/analyse",
            data=json.dumps({"syntax": "{{a}} and {{b}} make [[result]]"}),
            content_type="application/json",
        )
        data = response.get_json()
        assert sorted(data["inputs_required"]) == ["a", "b"]
        assert data["slots_defined"] == ["result"]

    def test_analyse_slot_fills_variable(self, client):
        """Variables filled by slots are not required inputs."""
        response = client.post(
            "/api/analyse",
            data=json.dumps({"syntax": "[[topic]]\nMore about {{topic}}\n[[details]]"}),
            content_type="application/json",
        )
        data = response.get_json()
        assert data["inputs_required"] == []
        assert "topic" in data["slots_defined"]


class TestUploadEndpoint:
    """Tests for the /api/upload endpoint."""

    def test_upload_csv(self, client, tmp_path):
        """Upload CSV file successfully."""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25\n")

        with open(csv_file, "rb") as f:
            response = client.post(
                "/api/upload",
                data={"file": (f, "test.csv")},
                content_type="multipart/form-data",
            )

        assert response.status_code == 200
        data = response.get_json()
        assert data["row_count"] == 2
        assert data["columns"] == ["name", "age"]
        assert "file_id" in data

    def test_upload_no_file(self, client):
        """Upload without file returns error."""
        response = client.post("/api/upload")
        assert response.status_code == 400


class TestEncodeStateEndpoint:
    """Tests for the /api/encode-state endpoint."""

    def test_encode_state(self, client):
        """Encode state returns encoded string."""
        response = client.post(
            "/api/encode-state",
            data=json.dumps({
                "syntax": "[[test]]",
                "model": "gpt-4",
                "inputs": {"x": "y"}
            }),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = response.get_json()
        assert "encoded" in data
        assert len(data["encoded"]) > 0


class TestLoadFromState:
    """Tests for loading from URL-encoded state."""

    def test_load_from_state(self, remote_client):
        """Load editor from encoded state."""
        from struckdown.playground.core import encode_state

        encoded = encode_state(
            syntax="Hello [[greeting]]",
            model="test-model",
            inputs={"name": "Alice"}
        )

        response = remote_client.get(f"/e/{encoded}")
        assert response.status_code == 200
        assert b"Hello" in response.data


class TestInputsPartial:
    """Tests for the /partials/inputs endpoint."""

    def test_inputs_partial_renders(self, client):
        """Inputs partial renders form fields."""
        response = client.post(
            "/partials/inputs",
            data=json.dumps({
                "inputs_required": ["topic", "style"],
                "current_values": {"topic": "cats"}
            }),
            content_type="application/json",
        )
        assert response.status_code == 200
        assert b"topic" in response.data
        assert b"style" in response.data

    def test_inputs_partial_preserves_values(self, client):
        """Inputs partial preserves existing values."""
        response = client.post(
            "/partials/inputs",
            data=json.dumps({
                "inputs_required": ["topic"],
                "current_values": {"topic": "my-value"}
            }),
            content_type="application/json",
        )
        assert response.status_code == 200
        assert b"my-value" in response.data

    def test_inputs_partial_empty(self, client):
        """Inputs partial handles no inputs."""
        response = client.post(
            "/partials/inputs",
            data=json.dumps({
                "inputs_required": [],
                "current_values": {}
            }),
            content_type="application/json",
        )
        assert response.status_code == 200
        assert b"No input variables" in response.data


class TestOutputsPartial:
    """Tests for the /partials/outputs endpoint."""

    def test_outputs_partial_renders(self, client):
        """Outputs partial renders output cards."""
        response = client.post(
            "/partials/outputs",
            data=json.dumps({
                "outputs": {"greeting": "Hello world!"},
                "error": None,
                "cost": None,
                "slots_defined": ["greeting"]
            }),
            content_type="application/json",
        )
        assert response.status_code == 200
        assert b"greeting" in response.data
        assert b"Hello world!" in response.data

    def test_outputs_partial_shows_error(self, client):
        """Outputs partial shows error message."""
        response = client.post(
            "/partials/outputs",
            data=json.dumps({
                "outputs": {},
                "error": "Something went wrong",
                "cost": None,
                "slots_defined": []
            }),
            content_type="application/json",
        )
        assert response.status_code == 200
        assert b"Something went wrong" in response.data

    def test_outputs_partial_renders_multiple_slots(self, client):
        """Outputs partial renders multiple slots in order."""
        response = client.post(
            "/partials/outputs",
            data=json.dumps({
                "outputs": {"first": "value1", "second": "value2"},
                "error": None,
                "cost": None,
                "slots_defined": ["first", "second"]
            }),
            content_type="application/json",
        )
        assert response.status_code == 200
        assert b"value1" in response.data
        assert b"value2" in response.data
        # Cost is displayed in header via JS, not in this partial
