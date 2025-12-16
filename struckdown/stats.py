"""Statistics calculation for comparing ground truth columns to completion outputs."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ColumnStats:
    """Statistics for a single column comparison."""

    column_name: str
    completion_name: str
    total_samples: int
    valid_samples: int
    matching_samples: int
    accuracy: float
    labels: list[str] = field(default_factory=list)
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    crosstab: list[dict] = field(default_factory=list)  # flat list for easy display
    precision_per_class: dict[str, float] = field(default_factory=dict)
    recall_per_class: dict[str, float] = field(default_factory=dict)
    f1_per_class: dict[str, float] = field(default_factory=dict)
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    # agreement metrics
    kappa: float = 0.0
    balanced_accuracy: float = 0.0
    mutual_information: float = 0.0
    normalised_mi: float = 0.0

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "column_name": self.column_name,
            "completion_name": self.completion_name,
            "total_samples": self.total_samples,
            "valid_samples": self.valid_samples,
            "matching_samples": self.matching_samples,
            "accuracy": self.accuracy,
            "labels": self.labels,
            "crosstab": self.crosstab,
            "confusion_matrix": self.confusion_matrix,
            "precision_per_class": self.precision_per_class,
            "recall_per_class": self.recall_per_class,
            "f1_per_class": self.f1_per_class,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "kappa": self.kappa,
            "balanced_accuracy": self.balanced_accuracy,
            "mutual_information": self.mutual_information,
            "normalised_mi": self.normalised_mi,
        }


def parse_compare_spec(spec: str) -> tuple[str, str]:
    """
    Parse a compare specification.

    "gender" -> ("gender", "gender")
    "sex=gender" -> ("sex", "gender")
    """
    if "=" in spec:
        column, completion = spec.split("=", 1)
        return (column.strip(), completion.strip())
    return (spec.strip(), spec.strip())


def normalise_value(value) -> str | None:
    """Normalise a value for comparison."""
    if value is None:
        return None
    s = str(value).strip().lower()
    return s if s else None


def build_confusion_matrix(
    ground_truth: list[str], predictions: list[str]
) -> tuple[list[str], dict[str, dict[str, int]]]:
    """
    Build confusion matrix from paired values.

    Returns (labels, matrix) where matrix[actual][predicted] = count
    """
    labels = sorted(set(ground_truth) | set(predictions))
    matrix = {actual: {pred: 0 for pred in labels} for actual in labels}

    for actual, pred in zip(ground_truth, predictions):
        matrix[actual][pred] += 1

    return labels, matrix


def build_crosstab(
    labels: list[str],
    matrix: dict[str, dict[str, int]],
    column_name: str,
    completion_name: str,
) -> list[dict]:
    """
    Build a flat crosstab list from confusion matrix.

    Returns list of dicts with column value, completion value, count, and match flag.
    """
    crosstab = []
    for actual in labels:
        for pred in labels:
            count = matrix[actual][pred]
            if count > 0:
                crosstab.append(
                    {
                        column_name: actual,
                        completion_name: pred,
                        "count": count,
                        "match": actual == pred,
                    }
                )
    return crosstab


def calculate_f1_metrics(
    labels: list[str], matrix: dict[str, dict[str, int]], min_n: int = 0
) -> tuple[dict[str, float], dict[str, float], dict[str, float], float, float]:
    """
    Calculate precision, recall, F1 per class, macro F1, weighted F1.

    Args:
        labels: list of unique labels
        matrix: confusion matrix as matrix[actual][predicted] = count
        min_n: minimum ground truth support to include in macro/weighted F1

    Returns (precision, recall, f1, macro_f1, weighted_f1)
    """
    precision = {}
    recall = {}
    f1 = {}
    support = {}

    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in labels if other != label)
        fn = sum(matrix[label][other] for other in labels if other != label)

        precision[label] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[label] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision[label] + recall[label] > 0:
            f1[label] = (
                2
                * precision[label]
                * recall[label]
                / (precision[label] + recall[label])
            )
        else:
            f1[label] = 0.0

        support[label] = sum(matrix[label].values())

    # only include labels with support >= min_n in aggregate metrics
    included_labels = [l for l in labels if support[l] >= min_n]

    if included_labels:
        macro_f1 = sum(f1[l] for l in included_labels) / len(included_labels)
    else:
        macro_f1 = 0.0

    total_support = sum(support[l] for l in included_labels)
    weighted_f1 = (
        sum(f1[label] * support[label] for label in included_labels) / total_support
        if total_support > 0
        else 0.0
    )

    return precision, recall, f1, macro_f1, weighted_f1


def calculate_agreement_metrics(
    labels: list[str], matrix: dict[str, dict[str, int]]
) -> tuple[float, float, float, float]:
    """
    Calculate Cohen's kappa, balanced accuracy, mutual information, and normalised MI.

    Args:
        labels: sorted list of unique labels
        matrix: confusion matrix as matrix[actual][predicted] = count

    Returns:
        (kappa, balanced_accuracy, mutual_information, normalised_mi)
    """
    # convert dict-of-dicts to numpy array
    C = np.array(
        [[matrix[actual][pred] for pred in labels] for actual in labels], dtype=float
    )
    N = C.sum()

    if N == 0:
        return 0.0, 0.0, 0.0, 0.0

    row_sums = C.sum(axis=1)
    col_sums = C.sum(axis=0)

    # Cohen's kappa
    observed_acc = np.trace(C) / N
    p_true = row_sums / N
    p_pred = col_sums / N
    expected_acc = (p_true * p_pred).sum()
    kappa = (
        (observed_acc - expected_acc) / (1 - expected_acc) if expected_acc < 1 else 1.0
    )

    # balanced accuracy (mean per-class recall)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall_per_class = np.diag(C) / row_sums
    recall_per_class = recall_per_class[~np.isnan(recall_per_class)]
    balanced_acc = recall_per_class.mean() if len(recall_per_class) > 0 else 0.0

    # mutual information
    P_ij = C / N
    P_i = row_sums / N
    P_j = col_sums / N

    # outer product for expected under independence
    P_expected = np.outer(P_i, P_j)

    # MI = sum P_ij * log2(P_ij / (P_i * P_j)), skip where P_ij == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log2(P_ij / P_expected)
    log_ratio = np.where(P_ij > 0, log_ratio, 0)
    MI = (P_ij * log_ratio).sum()

    # entropy of true labels for normalised MI
    with np.errstate(divide="ignore", invalid="ignore"):
        H_true = -(P_i * np.log2(P_i))
    H_true = np.where(P_i > 0, H_true, 0).sum()
    NMI = MI / H_true if H_true > 0 else 0.0

    return float(kappa), float(balanced_acc), float(MI), float(NMI)


def calculate_column_stats(
    ground_truth: list,
    predictions: list,
    column_name: str,
    completion_name: str,
    min_n: int = 1,
) -> ColumnStats:
    """
    Calculate confusion matrix and F1 scores for a column.

    Normalises values (lowercase, strip whitespace).
    Excludes None/empty values from calculations.

    Args:
        ground_truth: list of ground truth values
        predictions: list of predicted values
        column_name: name of the ground truth column
        completion_name: name of the completion slot
        min_n: minimum ground truth support to include label in aggregate metrics
    """
    total_samples = len(ground_truth)

    # normalise and pair values
    pairs = []
    for gt, pred in zip(ground_truth, predictions):
        gt_norm = normalise_value(gt)
        pred_norm = normalise_value(pred)
        if gt_norm is not None and pred_norm is not None:
            pairs.append((gt_norm, pred_norm))

    valid_samples = len(pairs)

    if valid_samples == 0:
        return ColumnStats(
            column_name=column_name,
            completion_name=completion_name,
            total_samples=total_samples,
            valid_samples=0,
            matching_samples=0,
            accuracy=0.0,
        )

    gt_values = [p[0] for p in pairs]
    pred_values = [p[1] for p in pairs]

    matching_samples = sum(1 for gt, pred in pairs if gt == pred)
    accuracy = matching_samples / valid_samples

    labels, matrix = build_confusion_matrix(gt_values, pred_values)
    precision, recall, f1, macro_f1, weighted_f1 = calculate_f1_metrics(
        labels, matrix, min_n
    )
    kappa, balanced_acc, mi, nmi = calculate_agreement_metrics(labels, matrix)
    crosstab = build_crosstab(labels, matrix, column_name, completion_name)

    return ColumnStats(
        column_name=column_name,
        completion_name=completion_name,
        total_samples=total_samples,
        valid_samples=valid_samples,
        matching_samples=matching_samples,
        accuracy=accuracy,
        labels=labels,
        confusion_matrix=matrix,
        crosstab=crosstab,
        precision_per_class=precision,
        recall_per_class=recall,
        f1_per_class=f1,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        kappa=kappa,
        balanced_accuracy=balanced_acc,
        mutual_information=mi,
        normalised_mi=nmi,
    )


def calculate_batch_stats(
    inputs: list[dict],
    results: list[dict],
    compare_specs: list[tuple[str, str]],
    min_n: int = 1,
) -> dict[str, ColumnStats]:
    """
    Calculate stats for each comparison specification.

    Args:
        inputs: list of input dicts (from spreadsheet rows)
        results: list of result dicts (completion outputs)
        compare_specs: list of (column_name, completion_name) tuples
        min_n: minimum ground truth support to include label in aggregate metrics

    Returns:
        dict mapping column_name -> ColumnStats
    """
    stats = {}
    for column_name, completion_name in compare_specs:
        ground_truth = [inp.get(column_name) for inp in inputs]
        predictions = [res.get(completion_name) for res in results]
        stats[column_name] = calculate_column_stats(
            ground_truth, predictions, column_name, completion_name, min_n
        )
    return stats


def format_stats_table(stats: dict[str, ColumnStats]) -> str:
    """Format statistics for terminal display."""
    if not stats:
        return ""

    lines = ["", "Comparison Statistics", "=" * 21, ""]

    for stat in stats.values():
        # header
        if stat.column_name == stat.completion_name:
            lines.append(f"{stat.column_name}")
        else:
            lines.append(
                f"{stat.column_name} (column) vs {stat.completion_name} (completion)"
            )
        lines.append("-" * len(lines[-1]))

        lines.append(f"Samples: {stat.total_samples} ({stat.valid_samples} valid)")
        lines.append(f"Accuracy: {stat.accuracy:.3f}")

        if stat.labels:
            # crosstab with row/column totals
            lines.append("")
            lines.append("Crosstab:")

            # calculate column widths
            value_width = max(len(str(label)) for label in stat.labels)
            value_width = max(value_width, 5)  # min width
            num_width = value_width + 2

            # row header label
            row_header = (
                f"data ({stat.column_name})"
                if stat.column_name != stat.completion_name
                else "actual"
            )
            col_header = (
                f"predicted ({stat.completion_name})"
                if stat.column_name != stat.completion_name
                else "predicted"
            )

            # calculate totals
            row_totals = {
                label: sum(stat.confusion_matrix[label].values())
                for label in stat.labels
            }
            col_totals = {
                label: sum(stat.confusion_matrix[row][label] for row in stat.labels)
                for label in stat.labels
            }
            grand_total = sum(row_totals.values())

            # build table with clear structure
            # Column header line
            data_width = num_width * (len(stat.labels) + 1)
            lines.append(f"  {'':<{value_width}} | {col_header:^{data_width}}")

            # Value labels row
            col_labels = (
                "".join(f"{label:>{num_width}}" for label in stat.labels)
                + f"{'total':>{num_width}}"
            )
            lines.append(f"  {'':<{value_width}} | {col_labels}")

            # separator
            lines.append(f"  {'-' * value_width}-+-{'-' * data_width}")

            # data rows
            for i, actual in enumerate(stat.labels):
                row_data = "".join(
                    f"{stat.confusion_matrix[actual][pred]:>{num_width}}"
                    for pred in stat.labels
                )
                row_label = row_header if i == 0 else ""
                lines.append(
                    f"  {actual:>{value_width}} | {row_data}{row_totals[actual]:>{num_width}}"
                )

            # separator and totals
            lines.append(f"  {'-' * value_width}-+-{'-' * data_width}")
            total_row = "".join(
                f"{col_totals[label]:>{num_width}}" for label in stat.labels
            )
            lines.append(
                f"  {'total':>{value_width}} | {total_row}{grand_total:>{num_width}}"
            )

            # Show row header label below table
            lines.append(f"  (rows: {row_header})")

            # per-class metrics
            lines.append("")
            lines.append("Per-class:")
            for label in stat.labels:
                p = stat.precision_per_class.get(label, 0)
                r = stat.recall_per_class.get(label, 0)
                f = stat.f1_per_class.get(label, 0)
                lines.append(f"  {label}: P={p:.3f} R={r:.3f} F1={f:.3f}")

            lines.append("")
            lines.append(f"Macro F1:    {stat.macro_f1:.3f}")
            lines.append(f"Weighted F1: {stat.weighted_f1:.3f}")

            lines.append("")
            lines.append(f"Cohen's κ:   {stat.kappa:.3f}")
            lines.append(f"Balanced Acc:{stat.balanced_accuracy:.3f}")
            lines.append(f"MI (bits):   {stat.mutual_information:.3f}")
            lines.append(f"Normalised MI:{stat.normalised_mi:.3f}")

        lines.append("")

    return "\n".join(lines)


def stats_to_json(stats: dict[str, ColumnStats]) -> dict:
    """Convert all stats to JSON-serializable format."""
    return {name: s.to_dict() for name, s in stats.items()}


def collect_error_examples(
    inputs: list[dict],
    results: list[dict],
    compare_specs: list[tuple[str, str]],
    max_per_type: int | None = None,
) -> dict[str, list[dict]]:
    """
    Collect examples of misclassifications.

    Args:
        inputs: list of input dicts
        results: list of result dicts
        compare_specs: list of (column_name, completion_name) tuples
        max_per_type: max examples per error type (None for all)

    Returns:
        dict mapping "column_name: actual -> predicted" to list of example dicts
    """
    errors: dict[str, list[dict]] = {}

    for column_name, completion_name in compare_specs:
        for i, (inp, res) in enumerate(zip(inputs, results)):
            actual = normalise_value(inp.get(column_name))
            predicted = normalise_value(res.get(completion_name))

            if actual is None or predicted is None:
                continue

            if actual != predicted:
                error_key = f"{column_name}: actual={actual}, predicted={predicted}"

                if error_key not in errors:
                    errors[error_key] = []

                # Check if we've hit the limit for this error type
                if max_per_type is not None and len(errors[error_key]) >= max_per_type:
                    continue

                # Build example with separate input and predicted sections
                example = {
                    "_index": i,
                    "_input_data": {},
                    "_predicted_data": {},
                }

                # Include input fields (excluding internal metadata)
                for k, v in inp.items():
                    if not k.startswith("_"):
                        example["_input_data"][k] = v

                # Only include completion slots in predicted section
                completion_slots = set(res.get("_completion_slots", []))
                for k, v in res.items():
                    if k in completion_slots:
                        example["_predicted_data"][k] = v

                errors[error_key].append(example)

    return errors


def format_error_examples(
    errors: dict[str, list[dict]], max_field_length: int = 500
) -> str:
    """Format error examples for terminal display."""
    if not errors:
        return ""

    def format_value(v, indent: str = "      ") -> str:
        """Format a value, truncating and indenting multiline."""
        v_str = str(v) if v is not None else ""
        if len(v_str) > max_field_length:
            v_str = v_str[:max_field_length] + "..."
        if "\n" in v_str:
            v_str = v_str.replace("\n", "\n" + indent)
        return v_str

    lines = ["", "Classification Errors", "=" * 21]

    for error_type, examples in errors.items():
        lines.append("")
        lines.append(f"{error_type} ({len(examples)} examples)")
        lines.append("-" * len(lines[-1]))

        for ex in examples:
            lines.append("")
            lines.append(f"  Row {ex.get('_index', '?')}:")

            # Show input data section
            input_data = ex.get("_input_data", {})
            if input_data:
                lines.append("    [.data]")
                for k, v in input_data.items():
                    lines.append(f"      {k}: {format_value(v, '        ')}")

            # Show predicted data section
            predicted_data = ex.get("_predicted_data", {})
            if predicted_data:
                lines.append("    [.predicted]")
                for k, v in predicted_data.items():
                    lines.append(f"      {k}: {format_value(v, '        ')}")

    lines.append("")
    return "\n".join(lines)
