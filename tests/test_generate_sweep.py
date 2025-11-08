import types

import json

# We will mock minimal tokenizer and model to simulate sweep flow without heavy deps.

class DummyTokenizer:
    eos_token_id = 0

    def __call__(self, inputs, padding=None, truncation=None, max_length=None, return_tensors=None):
        # map each input to a fake length; return dict with tensors-like lists
        n = len(inputs)
        # Represent input_ids/attention_mask as simple lists; generate.py only uses .shape or indexing when tensor-like.
        # We'll provide lists with .__getitem__ behavior.
        class _Arr(list):
            @property
            def shape(self):
                return (len(self), len(self[0]) if self else 0)
        ids = _Arr([[1] for _ in range(n)])
        att = _Arr([[1] for _ in range(n)])
        return {"input_ids": ids, "attention_mask": att}

    def decode(self, token):
        return "X"  # return constant token for predictability

    def batch_decode(self, tokens):
        return [self.decode(t) for t in tokens]

class DummyOutputs:
    def __init__(self, batch):
        # produce logits with simple pattern: higher values for last row to simulate selection
        import math
        import types as _t
        class _Arr(list):
            @property
            def shape(self):
                return (len(self), len(self[0]) if self else 0)
        # logits shape (batch, vocab=3)
        logits = _Arr([[ -1.0, -2.0, -3.0 ] for _ in range(batch)])
        if batch > 1:
            logits[-1] = [ -0.1, -0.2, -3.0 ]
        self.logits = _NamespaceLike(logits)
        self.past_key_values = None

class _NamespaceLike:
    def __init__(self, logits):
        self._logits = logits
    def __getitem__(self, k):
        return getattr(self, k)
    @property
    def logits(self):
        class _Last:
            def __init__(self, arr):
                self.arr = arr
            def __getitem__(self, item):
                return self.arr[item]
            def __sub__(self, other):
                return self
        return self._logits

class DummyModel:
    def __init__(self):
        self.device = "cpu"
    def __call__(self, input_ids=None, attention_mask=None, use_cache=None, past_key_values=None):
        batch = len(input_ids)
        return DummyOutputs(batch)

# Minimal smoke test for sampling-rates util parsing and sweep accumulation.

def test_sampling_rates_parse_and_sweep_monkeypatch(monkeypatch, tmp_path):
    from scripts import generate as G

    # Patch loader to return our dummy components
    monkeypatch.setattr(G, "load_model_and_tokenizer", lambda torch_dtype="auto": (DummyModel(), DummyTokenizer(), "cpu"))
    # Provide trivial contexts and task
    contexts = ["A", "B", "C"]
    # Monkeypatch context loader to bypass file IO
    monkeypatch.setattr(G, "load_contexts", lambda spec: contexts)

    # Build args namespace mimicking CLI
    args = types.SimpleNamespace(
        method="nbce",
        contexts="inline",
        task="要約",
        window_size=8,
        max_new_tokens=4,
        right_indentation=False,
        output=str(tmp_path / "agg.json"),
        dtype="auto",
        sampling_rates="0.2,0.3",
        merge_short=False,
        legacy_nbce=True,
        output_prefix=str(tmp_path / "result"),
    )

    # Run main flow partially: call internal functions
    rates = G._parse_sampling_rates(args.sampling_rates)
    assert rates == [0.2, 0.3]

    # Execute sweep loop body by calling main after monkeypatching parse_args
    monkeypatch.setattr(G, "parse_args", lambda: args)
    G.main()

    # Check output files
    with open(args.output, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "sweep" in data and len(data["sweep"]) == 2
