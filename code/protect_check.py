"""H31 保护机制强制检查器（SIMPLEX 专用）

每次启动长任务（训练/搜索/画图/重写/交付）前必须运行本脚本，
输出 PASS/FAIL 判定。任何 FAIL 必须解决后才能继续。

用法：
    python protect_check.py                  # 全部检查
    python protect_check.py --config-check   # 仅配置加载验证
    python protect_check.py --quick          # 快速检查（不查 GPU/进程）
"""
import json
import os
import sys

ROOT = r"C:\Users\TS\WorkBuddy\HydroGelNet"
TUNING = os.path.join(ROOT, "results", "tuning")
METRICS = os.path.join(ROOT, "results", "metrics")
PREDS = os.path.join(ROOT, "results", "preds")

FAILS = []
WARNS = []
PASSES = []


def check(name, ok, detail):
    (PASSES if ok else FAILS).append((name, detail))
    return ok


# ---- A. 输入验证：配置加载 ----
def config_check():
    sys.path.insert(0, os.path.join(ROOT, "code"))
    import paths  # noqa: E402
    cfg_p = paths.BEST_CONFIG_JSON
    if not os.path.exists(cfg_p):
        check("config exists", False, f"missing {cfg_p}")
        return
    with open(cfg_p, encoding="utf-8") as f:
        cfg = json.load(f)
    # 检查关键字段是否合理（保护：防止又加载错配置）
    swa = cfg.get("use_swa")
    ema = cfg.get("use_ema")
    d_model = cfg.get("d_model")
    marker = cfg.get("_inherited_v2")
    ok = isinstance(swa, bool) and isinstance(ema, bool)
    check("config keys sane", ok, f"swa={swa} ema={ema} d_model={d_model} marker={marker}")
    # config_used（trainer 实际用的）对比
    cu_p = os.path.join(TUNING, "config_used.json")
    if os.path.exists(cu_p):
        with open(cu_p, encoding="utf-8") as f:
            cu = json.load(f)
        if "use_swa" in cu and cu.get("use_swa") != swa:
            check("config_used matches", False,
                  f"BEST_CONFIG swa={swa} but config_used swa={cu.get('use_swa')} "
                  f"-> trainer would load WRONG config")
        else:
            check("config_used matches", True,
                  f"config_used swa={cu.get('use_swa')} == file swa={swa}")


# ---- B. 输出验证：产物一致性 ----
def output_check():
    cv = os.path.join(METRICS, "cv_outer.csv")
    if os.path.exists(cv):
        import pandas as pd  # noqa: E402
        d = pd.read_csv(cv)
        r2 = d["R2"].mean()
        check("cv_outer present", True, f"mean R2={r2:.4f} n={len(d)}")
        # 与 config_used 关联：如果 config_used swa=False 但 cv 显示高分，可疑
        cu_p = os.path.join(TUNING, "config_used.json")
        if os.path.exists(cu_p):
            with open(cu_p, encoding="utf-8") as f:
                cu = json.load(f)
            if cu.get("use_swa") is False and r2 > 0.79:
                check("cv vs config consistent", False,
                      "config swa=False but R2 high -> check which config ran")
            else:
                check("cv vs config consistent", True, f"R2={r2:.4f} swa={cu.get('use_swa')}")
    else:
        check("cv_outer present", False, "missing (train not run yet)")


# ---- C. 进度可观测 ----
def progress_check():
    pid_p = os.path.join(ROOT, "results", "opt_detached.pid")
    if os.path.exists(pid_p):
        with open(pid_p) as f:
            pid = f.read().strip()
        # 不杀进程，仅提示
        check("detached pid recorded", True, f"pid={pid}")
    else:
        check("detached pid recorded", False, "no pid file (long task not launched via launcher?)")


# ---- D. 写盘保护 ----
def write_guard_check():
    guard = os.path.join(ROOT, "code", "_runtime_guard.py")
    if os.path.exists(guard):
        with open(guard, encoding="utf-8") as f:
            src = f.read()
        check("runtime guard present", "to_csv" in src and "retry" in src,
              "_runtime_guard.py monkey-patch active")
    else:
        check("runtime guard present", False, "_runtime_guard.py missing")


def main():
    quick = "--quick" in sys.argv
    cfg_only = "--config-check" in sys.argv

    if cfg_only:
        config_check()
    else:
        config_check()
        if not quick:
            output_check()
            progress_check()
            write_guard_check()

    print(f"\n=== H31 PROTECT CHECK: {len(PASSES)} pass, {len(FAILS)} fail ===")
    for n, d in PASSES:
        print(f"  [PASS] {n}: {d}")
    for n, d in FAILS:
        print(f"  [FAIL] {n}: {d}")
    if FAILS:
        print("\n>>> BLOCKED: resolve FAIL items before launching long tasks <<<")
        return 1
    print("\n>>> OK: safe to proceed <<<")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
