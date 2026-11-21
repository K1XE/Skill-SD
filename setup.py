# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# setup.py is the fallback installation script when pyproject.toml does not work
import os
from pathlib import Path

from setuptools import find_packages, setup

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "verl/version/version")) as f:
    __version__ = f.read().strip()

install_requires = [
    "accelerate",
    "alfworld==0.4.2",
    "codetiming",
    "datasets",
    "dill",
    "hydra-core",
    "gymnasium==0.29.1",
    "numpy",
    "pandas",
    "peft",
    "pyarrow>=19.0.0",
    "ray[default]>=2.41.0,<=2.54.0",
    "torch>=2.6.0",
    "torchdata",
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0",
    "transformers<=4.57.3",
    "textworld==1.7.0",
    "wandb",
    "packaging>=20.0",
    "qwen-vl-utils[decord]",
]

extras_require = {
    "test": ["pytest"],
    "vllm": ["vllm==0.11.0", "flash-attn==2.7.4.post1"],
}


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="skill-sd",
    version=__version__,
    package_dir={"": "."},
    packages=find_packages(where="."),
    url="https://github.com/K1XE/Skill-SD",
    license="Apache 2.0",
    author="Skill-SD contributors",
    description="Skill-SD: Skill-Conditioned Self-Distillation for Multi-turn LLM Agents",
    install_requires=install_requires,
    extras_require=extras_require,
    package_data={
        "": ["version/*"],
        "verl": ["trainer/config/*.yaml"],
        "agent_system.environments.env_package.alfworld": ["configs/*.yaml"],
    },
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
