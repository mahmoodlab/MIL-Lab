from training.config import TrainConfig, TaskType

def test_train_config_metrics():
    # Test valid metrics
    for metric in ["auto", "kappa", "balanced_accuracy", "auc", "f1_macro", "accuracy"]:
        config = TrainConfig(early_stopping_metric=metric)
        assert config.early_stopping_metric == metric

def test_train_config_post_init():
    # Test task_type conversion
    config = TrainConfig(task_type="binary")
    assert config.task_type == TaskType.BINARY
