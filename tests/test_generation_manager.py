from model.wrappers.generation_manager import GenerationManager, NBCEStrategy, PCWStrategy

def test_generation_manager_instantiation():
    class DummyModel:
        pass

    class DummyTokenizer:
        eos_token_id = 0

    dummy_model = DummyModel()
    dummy_tokenizer = DummyTokenizer()
    nbce = NBCEStrategy(dummy_model, dummy_tokenizer, device="cpu", window_size=64)
    pcw = PCWStrategy(dummy_model, dummy_tokenizer, device="cpu", window_size=64)
    mgr = GenerationManager(nbce=nbce, pcw=pcw)
    assert mgr.nbce is not None and mgr.pcw is not None
