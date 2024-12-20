from .model import AskChartLlamaForCausalLM
from .model import MoEAskChartLlamaForCausalLM
from .model import MoEAskChartLlamaForCausalLM
import transformers
a, b, c = transformers.__version__.split('.')[:3]
if a == '4' and int(b) >= 36:
    from .model import AskChartPhiForCausalLM
    from .model import MoEAskChartPhiForCausalLM
