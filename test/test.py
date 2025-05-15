import sys
import os
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(WORK_DIR)
sys.path.append(os.path.join(WORK_DIR, 'third_party/Matcha-TTS'))
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging
import torchaudio
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', choices=['xpu', 'cuda', 'cpu'], default='xpu', help='Set device (default: xpu)')
parser.add_argument('-l', '--loglevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help='Set log level (default: INFO)')
parser.add_argument('-m', '--model-path', type=str, default=os.path.join(WORK_DIR, 'pretrained_models/CosyVoice2-0.5B'), help='Path to the model directory (default: pretrained_models/CosyVoice2-0.5B)')
parser.add_argument('-p', '--profile', action='store_true', help='Enable profiling (default: False)')
args = parser.parse_args()

user_device = args.device
model_path = args.model_path
is_profile = args.profile
logging.getLogger().setLevel(args.loglevel)

cosyvoice = CosyVoice2(model_path, load_jit=False, load_trt=False, fp16=False, use_flow_cache=False, device=user_device, profile=is_profile)

# NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
# zero_shot usage
prompt_speech_16k = load_wav(os.path.join(WORK_DIR, 'asset/zero_shot_prompt.wav'), 16000)
start_time = time.perf_counter()
for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_{}_{}.wav'.format(user_device, i), j['tts_speech'], cosyvoice.sample_rate)
end_time = time.perf_counter()
logging.debug('full {} test time: {}s'.format(user_device, end_time - start_time))
