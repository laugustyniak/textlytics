import pickle
from collections import namedtuple
from pathlib import Path
from typing import Callable, Dict

Metrics = namedtuple('Metrics', ['precision', 'recall', 'f1'])

MODEL_INFO_PARAMS = {
    'train_file': lambda f_name: Path(f_name).name,
    'eval': lambda results: str(results[1]['aspect']),
    'epoch': str,
    'external_embedding_model': str
}


def print_models_info(models_info_path: Path, model_info_params: Dict[str, Callable]):
    for model_info_path in models_info_path.glob('*'):
        with open(model_info_path.as_posix(), 'rb') as model_info_file:
            model_info = pickle.load(model_info_file)
            for info, fn in model_info_params.items():
                print(info + ': ' + fn(model_info[info]))
            print()


def get_model_metrics(models_info_path: Path) -> Dict[str, Metrics]:
    metrics = {}
    for model_info_path in models_info_path.glob('*'):
        with open(model_info_path.as_posix(), 'rb') as model_info_file:
            model_info = pickle.load(model_info_file)
            model_name = Path(model_info['test_file']).stem.replace('-test', '')
            evals = model_info['eval'][1]['aspect']
            metrics[model_name] = Metrics(*evals)
    return metrics


if __name__ == '__main__':
    metrics = get_model_metrics(
        Path('/home/lukasz/github/nlp/nlp-architect/examples/aspect_extraction/models-baseline'))
    pass
