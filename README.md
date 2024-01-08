# llm4healthcare

LLM for Healthcare Applications

## Usage

How I transfer the code of `AICare-baselines` to support `llm4healthcare` needs.

```bash
# directory structure initialization
git clone https://github.com/yhzhu99/AICare-baselines.git
rm -rf AICare-baselines/datasets/mimic-iii
rm -rf AICare-baselines/datasets/mimic-iv
ln -s /data/wangzx/llm4healthcare/datasets/tjh /data/wangzx/llm4healthcare/AICare-baselines/datasets
ln -s /data/wangzx/llm4healthcare/datasets/mimic-iv /data/wangzx/llm4healthcare/AICare-baselines/datasets
code AICare-baselines # switch to AICare-baselines workspace (I use VSCode)
```


