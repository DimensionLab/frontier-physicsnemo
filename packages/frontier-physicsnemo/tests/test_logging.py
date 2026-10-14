# SPDX-License-Identifier: Apache-2.0

from frontier_physicsnemo import FrontierLogger


def test_logger_namespaces_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    logger = FrontierLogger(project="p", name="n", config={"model": "domino"})
    logger.log_minibatch({"loss": 1.0}, step=0)
    logger.log_epoch({"cd_rel_err": 0.05}, epoch=1)
    logger.finish()
    files = list((tmp_path / ".frontier" / "p").glob("*.metrics.jsonl"))
    assert files and "train/loss" in files[0].read_text()
