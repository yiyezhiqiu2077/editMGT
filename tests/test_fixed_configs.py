from src.explicit_region.config import load_config


def test_formal_configs_bind_fixed_200k(monkeypatch):
    values={"EDITMGT_MODEL_ROOT":"/model","DERIVED_ROOT":"/derived","INTEREDIT_ROOT":"/inter",
            "MAGICBRUSH_ROOT":"/magic/train","MAGICBRUSH_DEV_ROOT":"/magic/dev","CRISPEDIT_ROOT":"/crisp",
            "SCALEEDIT_ROOT":"/scale","EDITMGT_OUTPUT_ROOT":"/output","TRANSLATOR_REVISION":"rev"}
    for key,value in values.items():monkeypatch.setenv(key,value)
    for experiment in ("e1","e2","e3","e4"):
        config=load_config(f"configs/train/cluster_8g_{experiment}.yaml")
        assert config["data"]["name"]=="fixed_200k"
        assert "weights" not in config["data"]
        expected_steps=31250 if experiment=="e3" else 6250
        assert config["max_optimizer_steps"]==config["scheduler_horizon_steps"]==expected_steps
        assert config["batch_per_gpu"]*config["gradient_accumulation"]*8==32
        assert config["validation"]["interval_steps"]==(3125 if experiment=="e3" else 625)
