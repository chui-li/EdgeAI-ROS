from huggingface_hub import snapshot_download
import os

# 你要下載的模型
repo_id = "google/gemma-3-1b-it"

# 下載到本機的資料夾
local_dir = os.path.expanduser("./gemma-3-1b-it")

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,   # 直接存完整檔案，比較不容易混亂
    resume_download=True
)

print(f"模型已下載到：{local_dir}")