from typing import List, Callable
from abc import ABC, abstractmethod

class PreprocessorInterface(ABC):
    @abstractmethod
    def __call__(self, x, y):
        pass

class GenericPreprocessor(PreprocessorInterface):
    def __init__(self, preprocessor_list: List[Callable]):
        self._preprocessor_list = preprocessor_list
    
    def __call__(self, x, y):
        for preprocess_step in self._preprocessor_list:
            x, y = preprocess_step(x, y)
        return x, y