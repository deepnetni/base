import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from funasr.auto.auto_model import AutoModel


model = AutoModel(
    # model="paraformer-zh",
    model="paraformer-en",
    # model_revision="v2.0.4",
    # vad_model="fsmn-vad",
    # vad_model_revision="v2.0.4",
    # punc_model="ct-punc-c",
    # punc_model_revision="v2.0.4",
    # spk_model="cam++", spk_model_revision="v2.0.2",
)
res = model.generate(
    input=f"{model.model_path}/example/asr_example.wav",
    batch_size_s=300,
)
print(res)
