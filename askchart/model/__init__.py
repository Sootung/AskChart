from .language_model.askchart_llama import AskChartLlamaForCausalLM, AskChartLlamaConfig
from .language_model.askchart_llama_moe import MoEAskChartLlamaForCausalLM, MoEAskChartLlamaConfig
import transformers
a, b, c = transformers.__version__.split('.')[:3]
if a == '4' and int(b) >= 36:
    from .language_model.askchart_phi import AskChartPhiForCausalLM, AskChartPhiConfig
    from .language_model.askchart_phi_moe import MoEAskChartPhiForCausalLM, MoEAskChartPhiConfig
