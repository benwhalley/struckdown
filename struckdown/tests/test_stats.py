"""Tests for statistics calculation module."""

import pytest

from struckdown.stats import (
    ColumnStats,
    build_confusion_matrix,
    build_crosstab,
    calculate_batch_stats,
    calculate_column_stats,
    calculate_f1_metrics,
    format_stats_table,
    normalise_value,
    parse_compare_spec,
    stats_to_json,
)


class TestParseCompareSpec:
    """Test compare specification parsing."""

    def test_simple_spec(self):
        assert parse_compare_spec("gender") == ("gender", "gender")

    def test_mapped_spec(self):
        assert parse_compare_spec("sex=gender") == ("sex", "gender")

    def test_whitespace_handling(self):
        assert parse_compare_spec("  gender  ") == ("gender", "gender")
        assert parse_compare_spec("  sex = gender  ") == ("sex", "gender")

    def test_equals_in_value(self):
        # only split on first equals
        assert parse_compare_spec("col=val=extra") == ("col", "val=extra")


class TestNormaliseValue:
    """Test value normalisation."""

    def test_string_lowercase(self):
        assert normalise_value("MALE") == "male"
        assert normalise_value("Female") == "female"

    def test_whitespace_strip(self):
        assert normalise_value("  male  ") == "male"

    def test_none_passthrough(self):
        assert normalise_value(None) is None

    def test_empty_string_to_none(self):
        assert normalise_value("") is None
        assert normalise_value("   ") is None

    def test_numeric_to_string(self):
        assert normalise_value(42) == "42"
        assert normalise_value(3.14) == "3.14"


class TestBuildConfusionMatrix:
    """Test confusion matrix construction."""

    def test_binary_classification(self):
        gt = ["m", "m", "f", "f"]
        pred = ["m", "f", "f", "f"]

        labels, matrix = build_confusion_matrix(gt, pred)

        assert labels == ["f", "m"]
        assert matrix["m"]["m"] == 1  # true positive for m
        assert matrix["m"]["f"] == 1  # false negative for m
        assert matrix["f"]["f"] == 2  # true positive for f
        assert matrix["f"]["m"] == 0  # false negative for f

    def test_multiclass(self):
        gt = ["a", "b", "c", "a"]
        pred = ["a", "b", "b", "c"]

        labels, matrix = build_confusion_matrix(gt, pred)

        assert labels == ["a", "b", "c"]
        assert matrix["a"]["a"] == 1
        assert matrix["a"]["c"] == 1
        assert matrix["b"]["b"] == 1
        assert matrix["c"]["b"] == 1

    def test_empty_input(self):
        labels, matrix = build_confusion_matrix([], [])
        assert labels == []
        assert matrix == {}

    def test_labels_from_both_lists(self):
        # prediction has label not in ground truth
        gt = ["a", "a"]
        pred = ["a", "b"]

        labels, matrix = build_confusion_matrix(gt, pred)

        assert "b" in labels
        assert matrix["a"]["b"] == 1


class TestBuildCrosstab:
    """Test crosstab building."""

    def test_basic_crosstab(self):
        labels = ["f", "m"]
        matrix = {"m": {"m": 4, "f": 1}, "f": {"m": 1, "f": 4}}

        crosstab = build_crosstab(labels, matrix, "gender", "predicted")

        # should have entries for all non-zero counts
        assert len(crosstab) == 4
        # check structure
        for entry in crosstab:
            assert "gender" in entry
            assert "predicted" in entry
            assert "count" in entry
            assert "match" in entry

    def test_crosstab_match_flag(self):
        labels = ["a", "b"]
        matrix = {"a": {"a": 3, "b": 1}, "b": {"a": 0, "b": 2}}

        crosstab = build_crosstab(labels, matrix, "col", "comp")

        matches = [e for e in crosstab if e["match"]]
        non_matches = [e for e in crosstab if not e["match"]]

        assert len(matches) == 2  # a->a and b->b
        assert len(non_matches) == 1  # a->b (b->a has count 0, excluded)

    def test_crosstab_excludes_zero_counts(self):
        labels = ["x", "y"]
        matrix = {"x": {"x": 5, "y": 0}, "y": {"x": 0, "y": 3}}

        crosstab = build_crosstab(labels, matrix, "col", "comp")

        # only non-zero entries
        assert len(crosstab) == 2
        counts = [e["count"] for e in crosstab]
        assert 0 not in counts


class TestCalculateF1Metrics:
    """Test F1 metric calculations."""

    def test_perfect_classification(self):
        labels = ["m", "f"]
        matrix = {"m": {"m": 5, "f": 0}, "f": {"m": 0, "f": 5}}

        precision, recall, f1, macro_f1, weighted_f1 = calculate_f1_metrics(labels, matrix)

        assert precision["m"] == 1.0
        assert precision["f"] == 1.0
        assert recall["m"] == 1.0
        assert recall["f"] == 1.0
        assert f1["m"] == 1.0
        assert f1["f"] == 1.0
        assert macro_f1 == 1.0
        assert weighted_f1 == 1.0

    def test_imperfect_classification(self):
        labels = ["m", "f"]
        # 8 correct m, 2 m predicted as f
        # 7 correct f, 3 f predicted as m
        matrix = {"m": {"m": 8, "f": 2}, "f": {"m": 3, "f": 7}}

        precision, recall, f1, macro_f1, weighted_f1 = calculate_f1_metrics(labels, matrix)

        # precision for m: 8 / (8 + 3) = 0.727...
        assert abs(precision["m"] - 8 / 11) < 0.001
        # recall for m: 8 / (8 + 2) = 0.8
        assert abs(recall["m"] - 0.8) < 0.001

        # precision for f: 7 / (7 + 2) = 0.778
        assert abs(precision["f"] - 7 / 9) < 0.001
        # recall for f: 7 / (7 + 3) = 0.7
        assert abs(recall["f"] - 0.7) < 0.001

    def test_zero_predictions_for_class(self):
        labels = ["a", "b"]
        # everything predicted as 'a'
        matrix = {"a": {"a": 5, "b": 0}, "b": {"a": 5, "b": 0}}

        precision, recall, f1, macro_f1, weighted_f1 = calculate_f1_metrics(labels, matrix)

        assert precision["b"] == 0.0  # no predictions of b
        assert recall["b"] == 0.0  # no correct b
        assert f1["b"] == 0.0


class TestCalculateColumnStats:
    """Test full column stats calculation."""

    def test_basic_stats(self):
        gt = ["m", "m", "f", "f", "m"]
        pred = ["m", "f", "f", "f", "m"]

        stats = calculate_column_stats(gt, pred, "gender", "gender")

        assert stats.column_name == "gender"
        assert stats.completion_name == "gender"
        assert stats.total_samples == 5
        assert stats.valid_samples == 5
        assert stats.matching_samples == 4
        assert abs(stats.accuracy - 0.8) < 0.001

    def test_with_missing_values(self):
        gt = ["m", None, "f", "", "m"]
        pred = ["m", "f", "f", "f", None]

        stats = calculate_column_stats(gt, pred, "gender", "gender")

        assert stats.total_samples == 5
        assert stats.valid_samples == 2  # only rows 0 and 2 have both values
        assert stats.matching_samples == 2

    def test_case_insensitive(self):
        gt = ["Male", "FEMALE", "male"]
        pred = ["male", "female", "MALE"]

        stats = calculate_column_stats(gt, pred, "gender", "gender")

        assert stats.accuracy == 1.0  # all match after normalisation

    def test_all_missing(self):
        gt = [None, None, ""]
        pred = [None, "", None]

        stats = calculate_column_stats(gt, pred, "col", "comp")

        assert stats.valid_samples == 0
        assert stats.accuracy == 0.0
        assert stats.labels == []

    def test_different_column_completion_names(self):
        gt = ["a", "b"]
        pred = ["a", "b"]

        stats = calculate_column_stats(gt, pred, "actual_col", "predicted_comp")

        assert stats.column_name == "actual_col"
        assert stats.completion_name == "predicted_comp"


class TestCalculateBatchStats:
    """Test batch stats calculation."""

    def test_single_comparison(self):
        inputs = [{"gender": "m"}, {"gender": "f"}, {"gender": "m"}]
        results = [{"gender": "m"}, {"gender": "f"}, {"gender": "f"}]
        specs = [("gender", "gender")]

        stats = calculate_batch_stats(inputs, results, specs)

        assert "gender" in stats
        assert stats["gender"].accuracy == pytest.approx(2 / 3)

    def test_multiple_comparisons(self):
        inputs = [
            {"gender": "m", "sentiment": "pos"},
            {"gender": "f", "sentiment": "neg"},
        ]
        results = [
            {"gender": "m", "sentiment": "pos"},
            {"gender": "m", "sentiment": "neg"},
        ]
        specs = [("gender", "gender"), ("sentiment", "sentiment")]

        stats = calculate_batch_stats(inputs, results, specs)

        assert len(stats) == 2
        assert stats["gender"].accuracy == 0.5
        assert stats["sentiment"].accuracy == 1.0

    def test_mapped_column(self):
        inputs = [{"actual_gender": "m"}, {"actual_gender": "f"}]
        results = [{"predicted": "m"}, {"predicted": "f"}]
        specs = [("actual_gender", "predicted")]

        stats = calculate_batch_stats(inputs, results, specs)

        assert "actual_gender" in stats
        assert stats["actual_gender"].completion_name == "predicted"
        assert stats["actual_gender"].accuracy == 1.0

    def test_missing_column(self):
        inputs = [{"other": "x"}]
        results = [{"gender": "m"}]
        specs = [("gender", "gender")]

        stats = calculate_batch_stats(inputs, results, specs)

        # should handle gracefully with no valid samples
        assert stats["gender"].valid_samples == 0


class TestFormatStatsTable:
    """Test terminal formatting."""

    def test_empty_stats(self):
        assert format_stats_table({}) == ""

    def test_basic_formatting(self):
        stats = {
            "gender": ColumnStats(
                column_name="gender",
                completion_name="gender",
                total_samples=10,
                valid_samples=10,
                matching_samples=8,
                accuracy=0.8,
                labels=["f", "m"],
                confusion_matrix={"m": {"m": 4, "f": 1}, "f": {"m": 1, "f": 4}},
                precision_per_class={"m": 0.8, "f": 0.8},
                recall_per_class={"m": 0.8, "f": 0.8},
                f1_per_class={"m": 0.8, "f": 0.8},
                macro_f1=0.8,
                weighted_f1=0.8,
            )
        }

        output = format_stats_table(stats)

        assert "Comparison Statistics" in output
        assert "gender" in output
        assert "Accuracy: 0.800" in output
        assert "Crosstab:" in output
        assert "total" in output  # row/column totals
        assert "Macro F1:    0.800" in output

    def test_mapped_column_header(self):
        stats = {
            "sex": ColumnStats(
                column_name="sex",
                completion_name="gender",
                total_samples=5,
                valid_samples=5,
                matching_samples=5,
                accuracy=1.0,
            )
        }

        output = format_stats_table(stats)

        assert "sex (column) vs gender (completion)" in output


class TestStatsToJson:
    """Test JSON conversion."""

    def test_conversion(self):
        stats = {
            "gender": ColumnStats(
                column_name="gender",
                completion_name="gender",
                total_samples=10,
                valid_samples=10,
                matching_samples=8,
                accuracy=0.8,
                labels=["f", "m"],
                confusion_matrix={"m": {"m": 4, "f": 1}, "f": {"m": 1, "f": 4}},
                precision_per_class={"m": 0.8, "f": 0.8},
                recall_per_class={"m": 0.8, "f": 0.8},
                f1_per_class={"m": 0.8, "f": 0.8},
                macro_f1=0.8,
                weighted_f1=0.8,
            )
        }

        result = stats_to_json(stats)

        assert result["gender"]["column_name"] == "gender"
        assert result["gender"]["accuracy"] == 0.8
        assert result["gender"]["labels"] == ["f", "m"]
