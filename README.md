# Optivoy Optimizer

独立项目目录：`optivoy-optimizer`

当前版本已按需求 v1.5 收敛为三阶段架构：

1. Phase 1：地理聚类分天
2. Phase 2：按平均驾车时间分配酒店并优先最少换酒店
3. Phase 3：日内排序与时间窗校验

求解会执行 `2` 轮迭代：第一轮先生成初解，第二轮将 Phase 3 的真实排序结果反馈给 Phase 1/2，进一步修正分天与酒店分配。

不再把机场或餐馆作为规划节点。

## 本地运行

```bash
cd /Users/code-2026/optivoy.top/optivoy-optimizer

uvicorn app:app --host 0.0.0.0 --port 8088 --reload
curl http://127.0.0.1:8088/health
curl -X POST http://127.0.0.1:8088/solve \
  -H 'Content-Type: application/json' \
  --data @sample-request.json
```

## API

- `GET /health`
- `POST /solve`

## `/solve` 输入字段

请求级字段：

- `startDate`
- `paceMode`
- `hotelStrategy`
- `mealPolicy`
- `transportPreference`
- `maxIntradayDrivingMinutes`

点位 `points[]`：

- `pointType`
- `suggestedDurationMinutes`
- `latitude` / `longitude`
- `arrivalAnchor` / `departureAnchor`
- `openingHoursJson`
- `specialClosureDates`
- `lastEntryTime`
- `hasFoodCourt`
- `queueProfileJson`

酒店 `hotels[]`：

- `latitude` / `longitude`
- `arrivalAnchor` / `departureAnchor`
- `checkInTime` / `checkOutTime`

矩阵 `distanceMatrix.rows[]`：

- `transitMinutes`
- `drivingMinutes`
- `walkingMeters`
- `walkingMinutes`
- `distanceKm`
- `transitSummary`

## 求解逻辑

### Phase 1：地理聚类分天

- 基于驾车时间选种子点与聚类扩展
- 步行距离 `<= 1500m` 时优先使用步行距离判断近邻
- 每天最多 `8` 个点
- 每天容量不再按固定 `30` 分钟交通开销估算
- 每次尝试加点时，会用距离矩阵估算“酒店出发 -> 点位序列 -> 回酒店”的真实交通分钟数
- 每天可装载量按“点位停留 + 排队 + 真实交通 + 必要时 `45` 分钟午餐占位”联合判断

### Phase 2：分配酒店

- `single`：整趟行程选一间平均往返驾车时间最小的酒店
- `smart`：默认沿用当前酒店，仅当某天平均单程驾车时间 `> 60` 分钟且存在另一家酒店能再减少 `30` 分钟以上时才切换
- 第二轮会优先参考上一轮的酒店分配结果，再结合新分天结果重新计算

### Phase 3：日内排序

- 每天最多 `8` 个点，因此始终全排列枚举精确顺序
- 严格检查：
  - 营业时间
  - 特殊闭店日期
  - 最晚入场
  - 返回酒店是否仍在日程时间窗内

## 餐饮规则

- `mealPolicy=auto` 时：
  - 当天存在 `hasFoodCourt=true` 点位，不额外扣午餐时间
  - 当天存在时长 `>= 240` 分钟点位，也视为可在点位内解决午餐
  - 否则从当天容量中扣除 `45` 分钟，并在路线模拟中插入午餐占位
- 如果启用午餐占位后仍然排不下，会自动用 `mealPolicy=off` 再试一次

## diagnostics

返回至少包含：

- `optimizationRounds`
- `totalTravelMinutes`
- `totalQueueMinutes`
- `totalLunchBreakMinutes`
- `fallbackEdgesUsed`
- `hotelSwitches`
- 每日：
  - `pointIds`
  - `hotelId`
  - `travelMinutes`
  - `queueMinutes`
  - `lunchBreakMinutes`
  - `transportModes`
  - `windowWaitMinutes`

## 当前限制

- 仍是启发式三阶段求解，不是完整全局联合优化。
- 若矩阵缺边，会继续使用 fallback 估算，并在 diagnostics 中记录使用量。
