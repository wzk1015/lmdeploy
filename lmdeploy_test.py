# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
# os.environ["DEBUG_IDX"] = "0"
# import torch
# torch.autograd.set_detect_anomaly(True)

from lmdeploy import PytorchEngineConfig, pipeline
from lmdeploy.vl import load_image

print('begin')
backend_config = PytorchEngineConfig(
    max_batch_size=32,
    enable_prefix_caching=True,
    cache_max_entry_count=0.8,
    session_len=8192,
)

# image = load_image(
#  'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'
# )
image = load_image('https://huggingface.co/OpenGVLab/'
                   'Mono-InternVL-2B/resolve/main/images/fig2.jpg')

# pipe = pipeline('OpenGVLab/Mono-InternVL-2B', backend_config=backend_config)

# response = pipe(('describe this image', image))

# print('Mono-InternVL-2B', response.text)

pipe2 = pipeline(
    '/mnt/petrelfs/share_data/wangwenhai/internvl/release/InternVL2-2B',
    backend_config=backend_config)

response2 = pipe2(('describe this image', image))

print('InternVL2-2B', response2.text)

# response = pipe.stream_infer((f'describe this image', image))
# for item in response:
#     print(item.text, flush=True, end="")
# print("\n", flush=True)
