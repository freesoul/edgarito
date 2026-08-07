from edgarito.schemas.valuation.specialized import SpecializedValuationExtraction


class SpecializedExtractionConsolePresenter:
    def render(self, extraction: SpecializedValuationExtraction) -> str:
        identifier = extraction.ticker or f"CIK {extraction.company_id}"
        lines = [
            f"{identifier} - {extraction.company_name}",
            f"Extractor: {extraction.input_type.label} | "
            f"Readiness: {extraction.readiness.value}",
            f"Source scope: {extraction.source_scope}",
        ]
        if extraction.fields:
            lines.extend(
                [
                    "",
                    "EXTRACTED FIELDS",
                    f"{'Period':<17} {'Field':<43} {'Value':>18} {'Unit':<10} Origin",
                    "-" * 100,
                ]
            )
            for field in extraction.fields:
                period_kind = {
                    "annual": "",
                    "quarterly": "quarter",
                    "year_to_date": "YTD",
                    "instant": "instant",
                }[field.period_kind.value]
                period_label = (
                    f"FY{field.fiscal_year}"
                    if field.fiscal_period == "FY"
                    else f"FY{field.fiscal_year} {field.fiscal_period} {period_kind}"
                )
                lines.append(
                    f"{period_label:<17} {field.name:<43} "
                    f"{field.value:>18,.2f} {field.unit:<10} {field.origin.value}"
                )
                lines.append(f"        Source: {', '.join(field.source_concepts)}")
                if field.derivation:
                    lines.append(f"        Formula: {field.derivation}")
        else:
            lines.extend(["", "No supported standard facts were extracted."])

        if extraction.missing_inputs:
            lines.extend(["", "STILL REQUIRED"])
            lines.extend(f"- {item}" for item in extraction.missing_inputs)
        if extraction.limitations:
            lines.extend(["", "LIMITATIONS"])
            lines.extend(f"- {item}" for item in extraction.limitations)
        return "\n".join(lines)

