"""Compatibility imports for callers migrating to src.model_runner."""

from src.model_runner import (  # noqa: F401
    FakeModelRunner,
    HuggingFaceModelRunner,
    ModelRunner,
    SemanticVerifyInput,
    VerificationResult,
    build_model_runner,
)

FakeSemanticOracle = FakeModelRunner
HuggingFaceSemanticOracle = HuggingFaceModelRunner
SemanticOracle = ModelRunner


def build_oracle(config, use_fake_oracle: bool = False):
    return build_model_runner(config, use_fake_model_runner=use_fake_oracle)

FakeDSSDOracle = FakeModelRunner
HuggingFaceDSSDOracle = HuggingFaceModelRunner
DSSDOracle = ModelRunner
