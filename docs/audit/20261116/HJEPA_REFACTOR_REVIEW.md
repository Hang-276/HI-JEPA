# H-JEPA 标准目录迁移的独立复核

日期：2026-11-16。先只读比对归档交付版与新入口，进行AST和SHA256核查；随后按主代理要求，仅修复新wrapper的轻量帮助入口。未安装依赖、下载权重、生成数据、启动训练、提取特征或重算实验结果。执行入口时只传入 `--help`，保留初次失败与修复后通过的记录。

## 结论

**未发现科学计算或配置被改变。** 当前迁移限定在包导入、目录位置和源码快照保存。原服务器交付包保持独立归档，新入口不能冒充正在服务器运行的旧源码。

最初发现新wrapper在缺少科学依赖的Python中无法显示帮助，导入 `h5py` 时失败；该入口问题已按限定范围修复。修复后帮助退出码为0，未导入任何科学依赖。科学主模块及原交付包未因这次帮助修复改变。详见第4节。

## 1. 原交付版保存情况

- 原包：`archive/releases/hjepa_slow_selection_20261115/`，保留嵌套Git，状态为干净的 `main...origin/main`。
- 原run.py SHA256仍为 `0e6abc8d92d2193d42c4471cf8790dd547a5448064399db683e9145b9758ac3d`。
- 旧路径 `experiments/hjepa_slow_selection` 作为兼容入口，不能作为新服务器结果已经生成的证据。

## 2. 科学逻辑与配置对照

对原 `run.py` 与新 `src/slowrepr/fourroom_hjepa.py` 的函数AST比对，以下10个函数的结构完全相同：`save_json`、`sha256`、`collect`、`split_frames`、`states`、`load_backbone`、`encode`、`make_probe`、`fit_probe`、`predict`。

仅 `execute`、`main` 改变。逐段差异为：

| 内容 | 改动 | 科学影响 |
|---|---|---|
| 模块导入 | `metrics` 改为包内 `.metrics` | 无评分规则变化 |
| 根目录 | 从独立实验目录改为仓库根 | 仅定位变化 |
| 输入缓存 | 移到 `assets/fourroom_hjepa/` | 上游commit、权重名称、数据采集参数不变 |
| 配置 | 移到 `configs/fourroom_hjepa/default.json` | 文件字节一致 |
| 输出 | 移到 `results/fourroom_hjepa/<name>/` | 单次结果仍集中保存，独占配置创建逻辑不变 |
| 源码快照 | 从平铺文件改为保留仓库相对目录，增加script_receipt | 增加可追溯性，不改变科学计算 |

`src/slowrepr/metrics.py` 与原metrics逐字节一致，SHA256为 `ede7d23d218789f9ff66a362fed128d34a5290f4ace907bf7771fed51d219c8a`。

`configs/fourroom_hjepa/default.json` 与原config逐字节一致，SHA256为 `a35678c9ce7ac22add97b476adec604d5f9a2a791e497d649a89c5b9d2acb761`。

因此数据划分、种子、T193、lag4/16/32、两层同帧编码、读出结构和优化预算、REF定义、CAL中位数、12项Bonferroni与CP规则均未因本次迁移改变。静态顺序仍为固定阈值→全部TEST预测→预测回执→读取TEST真值→评分。

## 3. 路径与源码快照

新 `ROOT = Path(__file__).resolve().parents[2]` 对 `src/slowrepr/fourroom_hjepa.py` 正确指向仓库根。setup脚本的 `../..` 同样指向仓库根；新环境为 `.venv-fourroom`，输入缓存与运行模块一致。

新运行将封存七项，审核时均存在：

1. `src/slowrepr/__init__.py`
2. `src/slowrepr/fourroom_hjepa.py`
3. `src/slowrepr/metrics.py`
4. `scripts/fourroom_hjepa/run.py`
5. `scripts/fourroom_hjepa/setup.sh`
6. `configs/fourroom_hjepa/default.json`
7. `docs/FOURROOM_HJEPA.md`

快照保留上述相对目录，`script_receipt.json`记录其SHA256。运行根另存有效config，包含冒烟覆盖值；正式输入provenance、环境版本、划分、阈值、预测回执和结果布局沿用原流程。包内没有新的CAL真值或latent导出，文档如实说明。

这些是静态路径和写入逻辑审核，未启动一个新运行来验证实际快照输出。

## 4. 帮助入口与依赖缺口

执行无模型计算的帮助命令后，当前本机Python返回：

```text
ModuleNotFoundError: No module named 'h5py'
```

原因是wrapper在解析 `--help` 前导入整个实验模块，触发h5py、hdf5plugin、NumPy和Torch依赖加载。静态代码显示，在依赖齐全时，argparse的help会在CUDA检查、文件输出、模型读取和实验执行之前退出；因此没有发现帮助会启动模型前向的风险。但本轮观察到的结果是帮助未成功显示，不能记为通过。

**修复与再次核查。** 按主代理要求，将同样的三个参数先放在轻量wrapper中解析，之后才导入科学主模块并调用原main；原main继续按原逻辑解析参数，避免改变科学主流程。两个parser的三个 `add_argument` 调用经AST比对完全相同，包括参数顺序、默认值、choices、required和action。归档原包保持不动。

相同系统Python执行 `PYTHONDONTWRITEBYTECODE=1 python3 scripts/fourroom_hjepa/run.py --help`，退出码0，正确显示 `--name`、`--device {cpu,cuda}` 和 `--smoke`。另以标准库runpy仅执行同一帮助路径并捕获SystemExit，检查 `sys.modules`：NumPy、SciPy、h5py、hdf5plugin、Torch和科学主模块均未加载。没有创建新运行、读取模型或执行评分。

setup依赖清单仍与原版相同，`stable_worldmodel`导入但未显式安装或锁定来源的问题没有被静默修补。`docs/FOURROOM_HJEPA.md`第2节已明确说明依赖缺口、干净环境未验证及不得擅自将新版本称作精确复现。当前没有安装日志或固定上游源码缓存，不能断言服务器一定失败，也不能宣称该缺口已解决。

## 5. 单元测试的边界

`tests/test_fourroom_metrics.py`相对原test_metrics只有路径导入调整，七个测试的测试体不变。它仅构造固定小数组，调用评分/统计函数；未读取封存结果、加载checkpoint、调用编码器或优化读出。已用AST检查语法，本子审核未执行这七项单测。

同时解析了新主模块、wrapper和metrics的AST，均通过。未使用全量测试发现命令，未导入旧科研模块。

## 6. 本次审核版本指纹

| 文件 | SHA256 |
|---|---|
| `src/slowrepr/fourroom_hjepa.py` | `4f964fff7d4e60a3eb77f5ecbbea13823064003fd314a4f51865074e15ae5503` |
| `scripts/fourroom_hjepa/run.py` | `be5d6b68bdb0496452f882d827c0c879a9a5c0c9a4feb2cf099c75d14cf150f5` |
| `scripts/fourroom_hjepa/setup.sh` | `960e07f783ffa479065996ea3823a211f5bd03a9fa08dedd87261b36885ae479` |
| `tests/test_fourroom_metrics.py` | `75f7df03227e7e2eba744bc5151c885c38d71288a3ab0de03df32e7dbe74bbba` |
| `docs/FOURROOM_HJEPA.md` | `aae57218f73bb18abfb8fa6d87f57cb994f485404c0ee15e550cbfcbb90d0a5e` |

上述表格保留首次审核版本指纹。轻量help修复后的新wrapper SHA256为 `69dddff42ac95429da425027d10e686dd6b46876b652d1e7ed03bdca792249ea`；科学主模块仍为 `4f964fff7d4e60a3eb77f5ecbbea13823064003fd314a4f51865074e15ae5503`。新运行的script_receipt会按实际wrapper字节记录，不复用旧指纹。
