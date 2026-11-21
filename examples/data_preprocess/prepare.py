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
"""Create placeholder rows for ALFWorld; observations and rewards come from the environment."""

import argparse
from pathlib import Path

from datasets import Dataset


def prepare_data(local_dir: str, train_data_size: int, val_data_size: int) -> None:
    output_dir = Path(local_dir).expanduser() / "text"
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, filename, size in (("train", "train.parquet", train_data_size), ("test", "test.parquet", val_data_size)):
        rows = [
            {
                "data_source": "text",
                "prompt": [{"role": "user", "content": ""}],
                "ability": "agent",
                "extra_info": {"split": split, "index": index},
            }
            for index in range(size)
        ]
        Dataset.from_list(rows).to_parquet(output_dir / filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="text", choices=["text"])
    parser.add_argument("--local_dir", default="~/data/verl-agent/")
    parser.add_argument("--train_data_size", default=16, type=int)
    parser.add_argument("--val_data_size", default=128, type=int)
    args = parser.parse_args()
    prepare_data(args.local_dir, args.train_data_size, args.val_data_size)
