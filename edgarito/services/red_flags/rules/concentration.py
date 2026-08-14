from edgarito.config.red_flags import RedFlagsConfiguration
from edgarito.schemas.red_flags import RedFlagCategory, RedFlagWarning


class _ConcentrationRules:
    def _check_concentration(
        self, configuration: RedFlagsConfiguration, warnings: list[RedFlagWarning]
    ) -> None:
        if not configuration.concentration.enabled:
            return
        self._warning(
            warnings,
            code="concentration_data_unavailable",
            category=RedFlagCategory.CONCENTRATION,
            message=(
                "Customer and segment concentration data is not represented in the "
                "normalized financial model; concentration rules were not evaluated."
            ),
            required=(),
        )
