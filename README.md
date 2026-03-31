# Optivoy Optimizer (OR-Tools CP-SAT)

独立项目目录：`optivoy-optimizer`

## 本地运行

```bash
cd /Users/code-2026/optivoy.top/optivoy-optimizer

# 先确认可访问 PyPI（可选）
env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
curl -I --max-time 8 https://pypi.org/simple/ortools/

docker buildx build --platform linux/amd64 -t optivoy-optimizer:prod --load .
docker compose up -d
curl http://127.0.0.1:8088/health
curl -X POST http://127.0.0.1:8088/solve \
  -H 'Content-Type: application/json' \
  --data @sample-request.json
```

如果 PyPI 不通，可加镜像源构建：

```bash
docker buildx build \
  --platform linux/amd64 \
  -t optivoy-optimizer:prod \
  --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
  --load .
```

## API

- `GET /health`
- `POST /solve`

`/solve` 输入含：

- 城市与省份（可选）
- 到达机场/离开机场代码（可选）
- 到达机场/离开机场坐标（可选）
- 到达时间
- 机场到达缓冲时间（`airportBufferMinutes`，默认 90，可选 60-120）
- 用户选点（景点/商城/饭店）
- 城市酒店列表
- 距离矩阵（`distanceMatrix.rows`，来自 `transit_cache`）
- 三档强度模式（`light/standard/compact`）
- 酒店策略（单酒店或多酒店）
- 午餐策略（`auto/off`）
- 优化目标（`min_days` / `min_transit` / `min_days_then_transit`）

## 说明

- OR-Tools CP-SAT 在这里作为“确定性优化引擎”。
- 业务约束需要通过请求参数显式传入，不能靠镜像配置自动生成。
- 模型是否“全局最优”取决于你建模的目标函数与约束，以及求解状态是否 `OPTIMAL`。
- `min_days_then_transit` 采用字典序优化：先最少天数，再在最少天数约束下最小化通勤代理目标。
- `airportBufferMinutes` 会直接影响首日可用预算；当到达较晚时，模型会自动允许“首日不游玩、次日开始”。
- 若 `distanceMatrix` 存在对应边，模型优先使用缓存通勤时间；缺失边自动回退坐标估算。
