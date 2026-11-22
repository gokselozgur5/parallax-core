import parallax.dora as dora
import parallax.muon as muon
import parallax.optim as optim
import parallax.qgalore as qgalore


def test_schedulers_named_in_optim_docstring_exist():
    for name in ("CosineAnnealingWithWarmup", "LinearWarmup"):
        if name in (optim.__doc__ or ""):
            assert hasattr(optim, name), f"{name} is documented but not defined"


def test_standalone_contracts_match_the_trainer_adapters():
    assert not (hasattr(muon, "MuonOptimizer") and "not wired to parallax.trainer.Trainer" in (muon.__doc__ or ""))
    assert not (hasattr(qgalore, "QGaLoreAdamW") and "Not integrated into parallax.optim.AdamW" in (qgalore.__doc__ or ""))
    assert not (hasattr(dora, "DoRAModule") and "does not\n    subclass parallax.nn.Module" in (dora.__doc__ or ""))
