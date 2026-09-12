"""Validation reporting functionality."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import structlog

from ml4t.data.validation.base import Severity, ValidationResult

logger = structlog.get_logger()


@dataclass
class ValidationReport:
    """Comprehensive validation report."""

    symbol: str
    provider: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    results: list[ValidationResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_issues(self) -> int:
        """Total number of issues across all results."""
        return sum(len(result.issues) for result in self.results)

    @property
    def critical_count(self) -> int:
        """Total critical issues."""
        return sum(result.critical_count for result in self.results)

    @property
    def error_count(self) -> int:
        """Total error issues."""
        return sum(result.error_count for result in self.results)

    @property
    def warning_count(self) -> int:
        """Total warning issues."""
        return sum(result.warning_count for result in self.results)

    @property
    def passed(self) -> bool:
        """Whether all validations passed."""
        return all(result.passed for result in self.results)

    def add_result(self, result: ValidationResult, validator_name: str) -> None:
        """Add a validation result to the report."""
        result.metadata["validator"] = validator_name
        self.results.append(result)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "symbol": self.symbol,
            "provider": self.provider,
            "timestamp": self.timestamp.isoformat(),
            "passed": self.passed,
            "summary": {
                "total_issues": self.total_issues,
                "critical": self.critical_count,
                "errors": self.error_count,
                "warnings": self.warning_count,
            },
            "results": [
                {
                    "validator": r.metadata.get("validator", "unknown"),
                    "passed": r.passed,
                    "duration_seconds": r.duration_seconds,
                    "issues": [
                        {
                            "severity": issue.severity.value,
                            "check": issue.check,
                            "message": issue.message,
                            "details": issue.details,
                            "row_count": issue.row_count,
                            "sample_rows": issue.sample_rows,
                        }
                        for issue in r.issues
                    ],
                    "metadata": r.metadata,
                }
                for r in self.results
            ],
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: Path) -> None:
        """Save report to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())
        logger.info("Saved validation report", path=str(path))

    @classmethod
    def load(cls, path: Path) -> ValidationReport:
        """Load report from file."""
        with open(path) as f:
            data = json.load(f)

        report = cls(
            symbol=data["symbol"],
            provider=data["provider"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )

        # Reconstruct results
        for result_data in data["results"]:
            result = ValidationResult(
                passed=result_data["passed"],
                duration_seconds=result_data.get("duration_seconds"),
                metadata=result_data.get("metadata", {}),
            )

            # Reconstruct issues
            for issue_data in result_data["issues"]:
                from ml4t.data.validation.base import ValidationIssue

                issue = ValidationIssue(
                    severity=Severity(issue_data["severity"]),
                    check=issue_data["check"],
                    message=issue_data["message"],
                    details=issue_data.get("details", {}),
                    row_count=issue_data.get("row_count"),
                    sample_rows=issue_data.get("sample_rows"),
                )
                result.issues.append(issue)

            report.results.append(result)

        return report

    def print_summary(self) -> None:
        """Print a human-readable summary of the report."""
        print(f"\n{'=' * 60}")
        print(f"Validation Report for {self.symbol} ({self.provider})")
        print(f"{'=' * 60}")
        print(f"Timestamp: {self.timestamp.isoformat()}")
        print(f"Status: {'✅ PASSED' if self.passed else '❌ FAILED'}")
        print("\nSummary:")
        print(f"  Total Issues: {self.total_issues}")
        print(f"  Critical: {self.critical_count}")
        print(f"  Errors: {self.error_count}")
        print(f"  Warnings: {self.warning_count}")

        if self.results:
            print("\nValidation Results:")
            for result in self.results:
                validator_name = result.metadata.get("validator", "Unknown")
                status = "✅" if result.passed else "❌"
                print(f"\n  {status} {validator_name}:")

                if result.duration_seconds:
                    print(f"    Duration: {result.duration_seconds:.2f}s")

                if result.issues:
                    print(f"    Issues ({len(result.issues)}):")
                    for issue in result.issues[:5]:  # Show first 5 issues
                        icon = self._get_severity_icon(issue.severity)
                        print(f"      {icon} [{issue.check}] {issue.message}")
                        if issue.row_count:
                            print(f"         Affected rows: {issue.row_count}")

                    if len(result.issues) > 5:
                        print(f"      ... and {len(result.issues) - 5} more issues")
                else:
                    print("    No issues found")

        print(f"\n{'=' * 60}\n")

    def _get_severity_icon(self, severity: Severity) -> str:
        """Get icon for severity level."""
        icons = {
            Severity.INFO: "ℹ️",
            Severity.WARNING: "⚠️",
            Severity.ERROR: "❌",
            Severity.CRITICAL: "🚨",
        }
        return icons.get(severity, "•")

    def to_dataframe(self) -> pl.DataFrame:
        """Convert issues to a DataFrame for analysis."""
        rows = []
        for result in self.results:
            validator = result.metadata.get("validator", "unknown")
            for issue in result.issues:
                rows.append(
                    {
                        "validator": validator,
                        "severity": issue.severity.value,
                        "check": issue.check,
                        "message": issue.message,
                        "row_count": issue.row_count or 0,
                        "timestamp": issue.timestamp,
                    }
                )

        if not rows:
            # Return empty DataFrame with schema
            return pl.DataFrame(
                {
                    "validator": [],
                    "severity": [],
                    "check": [],
                    "message": [],
                    "row_count": [],
                    "timestamp": [],
                }
            )

        return pl.DataFrame(rows)
