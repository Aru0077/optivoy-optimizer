# Optivoy Optimizer

独立项目目录：`optivoy-optimizer`

当前版本已升级到 `v2.5` 管线：

1. Phase 0：城市特征自适应
2. Phase 1：分钟成本聚类分天
3. Phase 1.5：跨天局部搜索
4. Phase 2：OR-Tools 单天排序与酒店协同
5. Phase 3：两轮反馈收敛

求解会执行 `2` 轮迭代：第一轮先生成初解，第二轮将前一轮的真实分天、顺序、酒店结果反馈给聚类和酒店分配。

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

### Phase 0：城市特征自适应

- 基于 `city` 和点位分布自动识别城市 profile
- profile 会影响：
  - 步行阈值
  - 酒店切换阈值
  - 每日点位上限
  - 体力系数
  - 每日容量 buffer

### Phase 1：分钟成本聚类分天

- 步行可达点按 `walking_minutes * 0.7` 参与聚类权重，不再把“米”和“分钟”混比
- 每次尝试加点时，会估算“前序酒店出发 -> 点位序列 -> 当天酒店结束”的真实交通分钟数
- 每天容量按“停留 + 排队 + 真实交通 + 午餐占位”联合判断
- 每天点位上限由城市 profile 决定，默认不超过 `8`

### Phase 1.5：跨天局部搜索

- 先做 `Relocate`
- 再做 `Swap`
- 只接受容量满足且总路程至少改善 `5` 分钟的变动

### Phase 2：分配酒店

- `single`：整趟行程选一间平均往返驾车时间最小的酒店
- `smart`：先做向量化初分配，再由 OR-Tools 和严格时间窗校验挑出当日更优酒店
- 默认沿用当前酒店，仅当达到 profile 定义的切换触发阈值且改善量足够时才切换

### Phase 2：单天排序

- 优先用 OR-Tools Routing 求候选顺序
- 再用原有严格时间窗链路验真
- 若 OR-Tools 候选不可行，则降级到精确枚举 `solve_day_route`

### Phase 3：反馈收敛

- 第二轮复用上一轮的：
  - 点位分天
  - 点位顺序
  - 每日酒店
- 最终按 `totalTravel + hotelSwitchPenalty + singlePointPenalty + dayPenalty` 取最优轮次

## 餐饮规则

- `mealPolicy=auto` 时：
  - 当天存在 `hasFoodCourt=true` 点位，不额外扣午餐时间
  - 当天存在时长 `>= 240` 分钟点位，也视为可在点位内解决午餐
  - 否则从当天容量中扣除 `45` 分钟，并在路线模拟中插入午餐占位
- 如果启用午餐占位后仍然排不下，会自动用 `mealPolicy=off` 再试一次

## diagnostics

返回至少包含：

- `optimizationRounds`
- `cityProfile`
- `totalTravelMinutes`
- `totalQueueMinutes`
- `totalLunchBreakMinutes`
- `fallbackEdgesUsed`
- `hotelSwitches`
- `validationWarnings`
- 每日：
  - `pointIds`
  - `hotelId`
  - `travelMinutes`
  - `queueMinutes`
  - `lunchBreakMinutes`
  - `transportModes`
  - `windowWaitMinutes`

## 当前限制

- 仍是启发式分天 + 单天优化，不是完整全局联合优化。
- 若矩阵缺边，会继续使用 fallback 估算，并在 diagnostics 中记录使用量。
- `Routing` 阶段主要负责生成高质量候选顺序，最终正确性仍以严格时间窗验证为准。
