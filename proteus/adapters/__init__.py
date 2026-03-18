"""
proteus/adapters/__init__.py

Embedding extractor for SimpleFold, plugged into the FlowMatchingPredictor interface.

PROTEUS was developed and validated on SimpleFold only. The interface is designed
to be extensible, but compatibility with other predictors has not been tested.

Available adapters:
  SimpleFoldPredictor  — wraps SimpleFold (Apple Inc., MIT License)

Usage
-----
    from proteus.adapters.simplefold import SimpleFoldPredictor
    predictor = SimpleFoldPredictor.from_pretrained("/path/to/simplefold/weights")
    features = predictor.score("MKTAYIAKQRQISFVKSHFSRQ...", n_conformations=10)
"""
