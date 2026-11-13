"""Resume the phase 4 assembler with JSON-normalized configuration values."""

from src import generate_rve_dataset_parallel as generator
from src import rve_dataset


_original_config_payload = rve_dataset._config_payload


def _json_compatible_config_payload(config):
    payload = _original_config_payload(config)
    payload["compression_determinant_range"] = list(
        config.compression_determinant_range
    )
    return payload


# The existing run.json stores JSON arrays as lists. Normalize this tuple-valued
# field before the resume guard compares the in-memory run with that file.
rve_dataset._config_payload = _json_compatible_config_payload
generator._config_payload = _json_compatible_config_payload

if __name__ == "__main__":
    generator.main()
