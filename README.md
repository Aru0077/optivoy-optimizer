# Optivoy Optimizer

独立项目目录：`optivoy-optimizer`

当前版本默认使用“精确枚举”主路径：

1. 校验输入并规范化点位/酒店坐标
2. 构建 `LookupMaps + MatrixStore`
3. 识别 `cityProfile`
4. 递归枚举候选分天
5. 对每个候选天执行“酒店候选筛选 + 日内全排列 + 严格时间窗模拟”
6. 取全局总代价最小方案，返回 `OPTIMAL`

当点位数超过 `12` 个时，会自动降级到旧 `v2.5` 启发式管线作为兜底，避免超大输入导致求解时间失控。

不再把机场或餐馆作为规划节点，机场仅由 backend 用于生成机票链接。

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
- `staminaFactor`（景点级体力系数，默认 `1.0`）
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

### 精确枚举主路径

- `single`：先按全程点位向量化筛出若干候选酒店，再对每家酒店做全局分天搜索
- `smart`：搜索状态为“剩余点位集合 + 第几天 + 前一晚酒店”，每天先筛 `top-5` 酒店，再对点位顺序做全排列
- 每个日程组合都走严格时间窗模拟，检查：
  - 营业时间
  - 多时段营业窗口
  - 特殊闭馆日
  - 最晚入场
  - 排队时间
  - 午餐占位
  - 单日驾车分钟上限
  - 回酒店是否超出日窗
- 文档要求的两个旧问题已修复：
  - `lastEntryTime` 缺失时不再直接判死
  - `openingHoursJson` 为空时按“全天可访问”处理

### 旧管线兜底

- 超过精确求解规模阈值时，自动回退到原有 `v2.5`：
  - 城市特征自适应
  - 聚类分天
  - 局部搜索
  - OR-Tools / 精确枚举混合日内排序
  - 两轮反馈收敛

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

- 精确枚举主路径按当前业务规模设计，默认适配约 `10` 个点的真实用户输入。
- 若矩阵缺边，会继续使用 fallback 估算，并在 diagnostics 中记录使用量。
- 超过 `12` 个点时会回退到旧启发式，不保证 `OPTIMAL`。
