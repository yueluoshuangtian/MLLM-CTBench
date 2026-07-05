from .base import BaseCLearner
from .ewc import EWCLearner
from .lwf import LwFLearner
from .mas import MASLearner
from .OLoRA import OLoRALearner
from .tir import TIREWCLearner, TIRMASLearner
from .task_encoder import TaskEncoder
from .eproj import EprojLearner
from .der import DERLearner
from .LoTA import LoTALearner
from .L2P import L2PLearner
from .max_merge import MaxMergeLearner
# 下面两个依赖 CoIN/peft (与新版 peft API 不兼容)，按需懒加载，
# 避免无关方法因 CoIN ImportError 全部连带失败
try:
    from .ewclora import EWCLoraLearner
except ImportError:
    EWCLoraLearner = None
try:
    from .moelora import moeloraLearner
except ImportError:
    moeloraLearner = None
__ALL__ = ['BaseCLearner',
           'EWCLearner',
           'LwFLearner',
           'MASLearner',
           'OLoRALearner',
            'DERLearner',
           'TIREWCLearner',
           'TIRMASLearner',
           'EprojLearner',
           'TaskEncoder',
           'LoTALearner',
           'moeloraLearner',
           'L2PLearner',
           'MaxMergeLearner',
           'EWCLoraLearner']
