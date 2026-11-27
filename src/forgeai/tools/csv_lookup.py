import csv
import json
from pathlib import Path

from .registry import ToolResult


class CSVLookupTool:
    MAX_FILE_BYTES = 10 * 1024 * 1024
    MAX_SCANNED_ROWS = 100_000
    MAX_RETURNED_ROWS = 10
    MAX_CELL_CHARS = 500

    def execute(
        self,
        arguments: dict[str, object],
        allowed_files: dict[str, str],
    ) -> ToolResult:
        filename = arguments.get("file")
        column = arguments.get("column")
        value = arguments.get("value")
        requested_columns = arguments.get("result_columns")
        if not all(isinstance(item, str) and item for item in (filename, column)):
            return ToolResult(
                success=False,
                output="",
                error="Both 'file' and 'column' must be non-empty strings.",
            )
        if not isinstance(value, (str, int, float)):
            return ToolResult(
                success=False,
                output="",
                error="'value' must be a string or number.",
            )
        if requested_columns is not None and (
            not isinstance(requested_columns, list)
            or any(not isinstance(item, str) for item in requested_columns)
        ):
            return ToolResult(
                success=False,
                output="",
                error="'result_columns' must be a list of column names.",
            )

        allowed_path = allowed_files.get(filename)
        if allowed_path is None:
            return ToolResult(
                success=False,
                output="",
                error="The requested CSV is not among this agent's supplied documents.",
            )
        path = Path(allowed_path).resolve()
        try:
            if not path.is_file() or path.stat().st_size > self.MAX_FILE_BYTES:
                return ToolResult(
                    success=False,
                    output="",
                    error="The allowed CSV is missing or exceeds the size limit.",
                )
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fieldnames = reader.fieldnames or []
                if column not in fieldnames:
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Column '{column}' does not exist in the CSV.",
                    )
                result_columns = requested_columns or fieldnames
                unknown_columns = set(result_columns) - set(fieldnames)
                if unknown_columns:
                    return ToolResult(
                        success=False,
                        output="",
                        error="One or more requested result columns do not exist.",
                    )
                matches: list[dict[str, str]] = []
                for row_index, row in enumerate(reader):
                    if row_index >= self.MAX_SCANNED_ROWS:
                        break
                    if (row.get(column) or "").casefold() != str(value).casefold():
                        continue
                    matches.append(
                        {
                            key: (row.get(key) or "")[: self.MAX_CELL_CHARS]
                            for key in result_columns
                        }
                    )
                    if len(matches) >= self.MAX_RETURNED_ROWS:
                        break
        except (OSError, UnicodeError, csv.Error) as exc:
            return ToolResult(
                success=False,
                output="",
                error=f"Unable to read the supplied CSV: {exc}",
            )

        return ToolResult(
            success=True,
            output=json.dumps(matches, ensure_ascii=False),
        )
