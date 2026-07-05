# Install the packages in open-r1-multimodal .
cd src/open-r1-multimodal
pip install -e ".[dev]"

# Addtional modules
pip install wandb==0.18.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensorboardx -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install qwen_vl_utils torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install flash-attn --no-build-isolation

pip install transformers==4.49.0 -i https://pypi.tuna.tsinghua.edu.cn/simple# correct deepspeed support
pip install duckdb -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install math_verify==0.5.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install datasets -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install deepspeed -i https://pypi.tuna.tsinghua.edu.cn/simple