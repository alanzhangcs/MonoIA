from .monoia import build as _build_monoia

def build_monoia(cfg):
    assert cfg.get('type', None) is not None
    if cfg.get('type') == 'monoia':
        return _build_monoia(cfg)
    else:
        raise ValueError(f"Invalid model name: {cfg.get('type')}")