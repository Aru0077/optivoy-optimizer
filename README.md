# Optivoy Optimizer

独立项目目录：`optivoy-optimizer`

当前版本默认使用“精确枚举”主路径：

1. 校验输入并规范化点位/酒店坐标
2. 构建 `LookupMaps + MatrixStore`
3. 识别 `cityProfile`
4. 递归枚举候选分天与空白日分支
5. 对每个候选天执行“全量酒店排序 + 日内全排列 + 严格时间窗模拟”
6. 以“总日历跨度最少，其次总成本最小”为目标返回模型范围内最优解

其中次级成本现在按 `需求以及实现.md` v1.3 对齐为：

- `totalTravelMinutes`
- `hotelSwitchPenalty`
- `hotelCountPenalty`

也就是同跨度方案之间优先选择总交通更少、换酒店更少、使用酒店总数更少的方案。

当前请求规模按 `1–7` 天日历跨度、最多 `12` 个点位设计，不再依赖旧 `v2.5` 兜底作为对外契约的一部分。

不再把机场或餐馆作为规划节点，机场仅由 backend 用于生成机票链接。

## 本地运行

需要 Python `3.10+`。当前仓库内已额外验证过一套本地 `3.12` 环境。

```bash
cd /Users/code-2026/optivoy.top/optivoy-optimizer

uvicorn app:app --host 0.0.0.0 --port 8088 --reload
curl http://127.0.0.1:8088/health
curl -X POST http://127.0.0.1:8088/solve \
  -H 'Content-Type: application/json' \
  --data @sample-request.json
```

## 回归样例

已内置一组最小回归样例：

- `regression_cases/01-weekday-queue.json`
- `regression_cases/02-weekend-queue.json`
- `regression_cases/03-holiday-queue.json`
- `regression_cases/04-blank-day.json`
- `regression_cases/05-smart-switch.json`

执行方式：

```bash
cd /Users/code-2026/optivoy.top/optivoy-optimizer
python3 run_regressions.py
```

说明：

- 该脚本会直接导入 `app.py` 并调用 `solve_itinerary_exact`
- 运行前需要先安装 `requirements.txt` 中的依赖
- Python `3.9` 无法运行当前 optimizer，本地回归已在 `3.12` 环境验证通过
- 当前沙箱若缺少 `fastapi` / `ortools`，脚本会直接提示缺失依赖

## API

- `GET /health`
- `POST /solve`

## `/solve` 输入字段

请求级字段：

- `startDate`
- `calendarDays`
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

矩阵 `distanceMatrix.rows[]`：

- `transitMinutes`
- `drivingMinutes`
- `walkingMeters`
- `walkingMinutes`
- `distanceKm`
- `transitSummary`

## 求解逻辑

### 精确枚举主路径

- `single`：按全量酒店排序依次搜索固定酒店方案
- `smart`：搜索状态为“剩余点位集合 + 第几天 + 前一晚酒店 + 当前连住夜数 + 已使用酒店集合”，酒店排序只影响搜索顺序，不做 `top-k` 截断
- `smart` 搜索使用 branch-and-bound 剪枝：
  - 先用“剩余最少天数”下界过滤不可能更优的分支
  - 再用“酒店转移 + 首末点通勤”下界缩短单日酒店枚举
- `smart` 已实现的硬约束：
  - 非最后一段酒店至少连住 `2` 晚后才允许切换
  - 不同酒店总数不超过 `ceil(tripDays / 2)`
  - 酒店切换需满足当前酒店对剩余点位平均单程驾车时间超过阈值，且新酒店改善量达到阈值
- 每个日程组合都走严格时间窗模拟，检查：
  - 营业时间
  - 多时段营业窗口
  - 特殊闭馆日
  - 最晚入场
  - 排队时间
  - `calendarDays.dayType` 对 `holidayMinutes` / `weekendMinutes` / `weekdayMinutes` 的判定
  - 午餐占位
  - 单日驾车分钟上限
  - 换酒店通勤时间
  - 回酒店是否超出日窗
- 当某个日历日不存在任何可行非空子集时，允许生成空白日，并输出 `blankReason`
- 文档要求的两个旧问题已修复：
  - `lastEntryTime` 缺失时不再直接判死
  - `openingHoursJson` 为空时按“全天可访问”处理

## 餐饮规则

- `mealPolicy=auto` 时：
  - 当天存在 `hasFoodCourt=true` 点位，不额外扣午餐时间
  - 当天存在时长 `>= 240` 分钟点位，也视为可在点位内解决午餐
  - 否则从当天容量中扣除 `45` 分钟，并在路线模拟中插入午餐占位
- 如果启用午餐占位后仍然排不下，会自动用 `mealPolicy=off` 再试一次

## diagnostics

返回至少包含：

- `solverMode`
- `optimizationRounds`
- `cityProfile`
- `totalHotelsUsed`
- `totalTravelMinutes`
- `totalQueueMinutes`
- `totalLunchBreakMinutes`
- `fallbackEdgesUsed`
- `blankDays`
- `hotelSwitches`
- `validationWarnings`
- 每日：
  - `pointIds`
  - `hotelId`
  - `dayType`
  - `blankReason`
  - `hotelTransferMinutes`
  - `travelMinutes`
  - `queueMinutes`
  - `lunchBreakMinutes`
  - `transportModes`
  - `windowWaitMinutes`

`cityProfile` 中会额外返回 `hotelCountPenalty`，便于对照当前城市配置理解为什么 `single` 或 `smart` 会偏向更少酒店数。

## 当前限制

- `OPTIMAL` 仅表示当前模型、当前输入数据和当前酒店全集下的最优，不表示现实世界绝对最优。
- 精确枚举主路径按当前业务规模设计，默认适配最多 `12` 个点、最多 `7` 个日历日的真实用户输入。
- 若矩阵缺边，会继续使用 fallback 估算，并在 diagnostics 中记录使用量。
- `smart` 模式已取消 `top-k` 截断，并已接入 branch-and-bound 与酒店序列硬约束，但酒店数较多时仍需继续关注性能。
- backend 节假日日历当前按“稀疏覆盖”方式使用：只要某年份存在导入记录，未显式配置的普通日期会自然回退到 `weekday/weekend`，不会逐日告警。
