from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkConfig:
    data_path: Path
    output_dir: Path
    test_size: float = 0.2
    random_state: int = 42
    scale_features: bool = True

    def to_metadata(self) -> dict:
        payload = asdict(self)
        payload["data_path"] = str(self.data_path)
        payload["output_dir"] = str(self.output_dir)
        return payload

