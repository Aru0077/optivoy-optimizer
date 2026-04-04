# Optivoy Optimizer

独立项目目录：`optivoy-optimizer`

当前版本已按《优化器重构计划》切换为三阶段架构：

1. Phase 1：地理聚类分天
2. Phase 2：按每天点位质心分配酒店
3. Phase 3：日内排序与时间窗校验

不再使用 CP-SAT 联合求解分天与酒店。

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

- `arrivalDateTime`
- `arrivalAirport` / `departureAirport`
- `arrivalAirportBufferMinutes` / `departureAirportBufferMinutes`
- `paceMode`
- `hotelMode`
- `mealPolicy`
- `transportPreference`
- `maxDays`
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
- `mealSlots`
- `mealTimeWindowsJson`
- `queueProfileJson`

酒店 `hotels[]`：

- `latitude` / `longitude`
- `arrivalAnchor` / `departureAnchor`
- `foreignerFriendly`
- `checkInTime` / `checkOutTime`
- `bookingStatus`

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
- 每天容量不再只看停留时长
- 会扣除：
  - 点位间平均交通开销
  - 虚拟餐时间开销

### Phase 2：分配酒店

- `single`：整趟行程选一间平均往返驾车时间最小的酒店
- `multi`：每天独立选最近酒店

### Phase 3：日内排序

- `<= 8` 个点时枚举精确顺序
- `> 8` 个点时用最近邻生成顺序
- 严格检查：
  - 营业时间
  - 特殊闭店日期
  - 最晚入场
  - 最后一天机场缓冲

## 餐饮规则

- `mealPolicy=auto` 时：
  - `light` 只要求午餐
  - `standard/compact` 要求午餐和晚餐
- `hasFoodCourt=true` 的景点/商城可满足午餐和晚餐
- 虚拟餐失败不再直接报错，而是转成软惩罚
- 如果当天仍然排不下，会自动用 `mealPolicy=off` 再试一次

## diagnostics

返回至少包含：

- `totalTravelMinutes`
- `totalQueueMinutes`
- `totalVirtualMealMinutes`
- `totalMealPenaltyMinutes`
- `fallbackEdgesUsed`
- `hotelSwitches`
- 每日：
  - `pointIds`
  - `hotelId`
  - `travelMinutes`
  - `queueMinutes`
  - `virtualMealMinutes`
  - `mealPenaltyMinutes`
  - `transportModes`
  - `mealStatus`
  - `windowWaitMinutes`

## 当前限制

- Phase 1 分天仍以驾车时间作为聚类主依据，符合当前重构计划，但不是完整时空网络最优。
- `multi` 酒店模式当前按天独立选最近酒店，没有额外的跨天切换惩罚优化。
- 若矩阵缺边，会继续使用 fallback 估算，并在 diagnostics 中记录使用量。
