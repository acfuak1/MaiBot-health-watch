"""本地服务健康看护插件（local.health-watch）

定时探活本机依赖服务，异常时通过麦麦自己的账号发 QQ 告警；支持 `/health` 手动体检。

设计要点（沿用本机 外部看护脚本\maibot-watch.cjs 的实践经验）：
  1. **探活重试**：单次失败不算掉线（服务在加载模型/合成长句时会短暂不响应），
     重试 probe_attempts 次后仍失败才判定为 DOWN。
  2. **只在状态「由好变坏」时告警**，并按检查项做冷却，避免刷屏。
  3. **告警文案带修法**，收到就能直接照做。
  4. **不探测麦麦自身**：插件跑在麦麦的插件运行时里，宿主死了插件也死了 ——
     "麦麦自己挂了"必须交给外部通道（例如一个独立的计划任务 + 另一个机器人）。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import httpx

from maibot_sdk import Command, Field, MaiBotPlugin, PluginConfigBase

__all__ = ["HealthWatchPlugin", "create_plugin"]


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
class PluginMeta(PluginConfigBase):
    """`[plugin]` 节：宿主强制要求提供 config_version（版本策略）。"""

    enabled: bool = Field(default=True, description="是否启用插件")
    config_version: str = Field(default="1.0.0", description="配置版本（插件声明，勿手改）")
    version: str = Field(default="1.0.0", description="插件版本")


class HealthWatchConfig(PluginConfigBase):
    """插件配置（Runner 会据此生成 config.toml 并在 WebUI 中可视化编辑）。"""

    plugin: PluginMeta = Field(default_factory=PluginMeta)

    enabled: bool = Field(default=True, description="总开关：关掉后不再探活")
    interval_seconds: int = Field(default=300, description="探活间隔（秒），默认 5 分钟")
    cooldown_minutes: int = Field(default=60, description="同一检查项的告警冷却（分钟）")
    alert_user_id: str = Field(
        default="",
        description="告警接收人 QQ（私聊）。留空则只写日志、不发 QQ。",
    )
    alert_platform: str = Field(default="qq", description="告警平台标识")
    probe_attempts: int = Field(default=3, description="探活重试次数（全部失败才判定掉线）")
    probe_gap_seconds: float = Field(default=1.5, description="每次重试之间的间隔（秒）")

    # ── 日常使用：一般只需要填 alert_user_id，其余保持默认 ──
    auto_discover: bool = Field(
        default=True,
        description="自动发现本机在跑的常见服务（Ollama / 语音合成 / 语音识别 / WebUI 等）。"
        "开着就零配置：谁在跑盯谁，没装的服务永远不会报警。",
    )
    exclude_names: list[str] = Field(
        default_factory=list,
        description="不想盯着看的服务名（填一部分即可，例如填 语音 就会跳过所有带「语音」的）",
    )
    extra_targets: list[str] = Field(
        default_factory=list,
        description="进阶：额外要盯的服务，每行「显示名|地址[|修法提示[|判定方式]]」。"
        "不确定就别填。",
    )

# --------------------------------------------------------------------------- #
# 检查项定义
# --------------------------------------------------------------------------- #
# 内置的常见本地服务：**只发现、不假定**。
# 启动时先探一遍，谁能连上就盯谁；没装的（连不上）永远不会被报警 —— 所以零配置也能用。
# 想加自己的服务：见配置里的 extra_targets。
KNOWN_SERVICES = (
    # 名称, 探活地址, 判定方式（tts=要求 ok=true，ollama=要求有 models 列表，http=能连上即可）
    ("本地模型(Ollama)", "http://127.0.0.1:11434/api/tags", "ollama"),
    ("本地语音合成", "http://127.0.0.1:8720/health", "tts"),
    ("本地语音识别", "http://127.0.0.1:8910/", "http"),
    ("GPT-SoVITS 语音", "http://127.0.0.1:9880/", "http"),
    ("LM Studio", "http://127.0.0.1:1234/v1/models", "http"),
    ("vLLM / 本地推理", "http://127.0.0.1:8000/v1/models", "http"),
    ("ComfyUI", "http://127.0.0.1:8188/", "http"),
    ("Stable Diffusion", "http://127.0.0.1:7860/", "http"),
    ("one-api / new-api", "http://127.0.0.1:3000/", "http"),
)


class CheckSpec:
    """一个探活目标的描述。"""

    __slots__ = ("key", "label", "kind", "url", "timeout", "hint")

    def __init__(self, key: str, label: str, kind: str, url: str, timeout: float, hint: str):
        self.key = key
        self.label = label
        self.kind = kind        # "ollama" | "tts" | "http"
        self.url = url
        self.timeout = timeout
        self.hint = hint


class HealthWatchPlugin(MaiBotPlugin):
    """本地依赖服务的定时看护。"""

    config_model = HealthWatchConfig

    # 订阅全局配置热重载（可选）
    config_reload_subscriptions: ClassVar[Tuple[str, ...]] = ()

    def __init__(self) -> None:
        super().__init__()
        self._task: Optional[asyncio.Task] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._healthy: Dict[str, bool] = {}
        self._last_alert_at: Dict[str, float] = {}
        self._last_detail: Dict[str, str] = {}
        self._discovered: set = set()   # 曾成功连上过的常见服务名

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    async def on_load(self) -> None:
        """加载：读回持久化状态并启动定时循环。"""
        await self._restore_state()
        if bool(getattr(self.config, "auto_discover", True)):
            await self._discover()
        self._start_loop()
        self.ctx.logger.info(
            "健康看护已启动：间隔 %ss，告警接收人=%s",
            self.config.interval_seconds,
            self.config.alert_user_id or "(未配置，仅记日志)",
        )

    async def on_unload(self) -> None:
        """卸载：停掉循环并保存状态。"""
        await self._stop_loop()
        self._save_state()
        self.ctx.logger.info("健康看护已停止")

    async def on_config_update(self, scope: str, config_data: dict, version: str) -> None:
        """配置热更新：重启循环以应用新间隔与新目标。"""
        if scope != "self":
            return
        self.ctx.logger.info("健康看护配置已更新: version=%s", version)
        await self._stop_loop()
        self._start_loop()

    # ------------------------------------------------------------------ #
    # 循环
    # ------------------------------------------------------------------ #
    async def _discover(self) -> None:
        """探一遍内置的常见服务，记录「本机确实在跑」的那些。

        没连上的服务**不会被加入监视**，所以本机没装 Ollama / 没开语音服务时，
        永远不会收到关于它们的报警 —— 这是「零配置」能成立的关键。
        """
        before = set(self._discovered)
        found: List[str] = []
        # trust_env=False：绕开系统代理。否则 127.0.0.1 也会被代理接管，
        # 返回一个代理错误页 → 被误判成「服务在跑」（实测踩过这个坑）。
        async with httpx.AsyncClient(follow_redirects=True, trust_env=False) as client:
            for name, url, kind in KNOWN_SERVICES:
                probe = CheckSpec("probe", name, kind, url, 5.0, "")
                ok, _detail = await self._probe_once(client, probe)
                if ok:
                    self._discovered.add(name)
                    if name not in before:
                        found.append(name)
        if found:
            self.ctx.logger.info("发现本机在跑的服务：%s", "、".join(found))
            self._save_state()
        elif not self._discovered:
            self.ctx.logger.info(
                "未发现内置清单里的本地服务（本机没装或没启动）。想盯自己的服务，可在配置的 extra_targets 里添加。"
            )

    def _start_loop(self) -> None:
        if not bool(getattr(self.config, "enabled", True)):
            self.ctx.logger.info("健康看护被配置为停用，不启动循环")
            return
        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._loop(), name="health-watch-loop")

    async def _stop_loop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._stop_event = None

    async def _loop(self) -> None:
        # 启动后先等一会儿，别和宿主初始化抢资源
        await asyncio.sleep(20)
        while self._stop_event is not None and not self._stop_event.is_set():
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.ctx.logger.warning("健康看护本轮异常: %s", exc)
            try:
                interval = max(30, int(self.config.interval_seconds))
            except Exception:  # noqa: BLE001
                interval = 300
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    # ------------------------------------------------------------------ #
    # 探活
    # ------------------------------------------------------------------ #
    def _build_specs(self) -> List[CheckSpec]:
        """探活目标 = 已发现的常见服务 + 用户额外添加的（再按 exclude_names 排除）。"""
        specs: List[CheckSpec] = []
        exclude = [str(x).strip() for x in (getattr(self.config, "exclude_names", []) or []) if str(x).strip()]

        if bool(getattr(self.config, "auto_discover", True)):
            for name, url, kind in KNOWN_SERVICES:
                if any(token in name for token in exclude):
                    continue
                if name not in self._discovered:
                    # 从没连上过 —— 说明本机没装，不盯也不报警
                    continue
                specs.append(CheckSpec(f"known::{name}", name, kind, url, 10.0, ""))

        kind_map = {"json_ok": "tts", "ollama": "ollama", "http": "http"}
        for index, line in enumerate(getattr(self.config, "extra_targets", []) or []):
            text = str(line or "").strip()
            if not text or text.startswith("#"):
                continue
            parts = [piece.strip() for piece in text.split("|")]
            if len(parts) < 2 or not parts[1]:
                self.ctx.logger.warning("extra_targets 第 %d 行格式不对，已跳过：%s", index + 1, text[:80])
                continue
            label = parts[0] or f"自定义{index + 1}"
            if any(token in label for token in exclude):
                continue
            hint = parts[2] if len(parts) > 2 else ""
            kind = (parts[3] if len(parts) > 3 else "http").strip().lower()
            if kind not in kind_map:
                kind = "http"
            specs.append(CheckSpec(f"extra{index}", label, kind_map[kind], parts[1], 10.0, hint))
        return specs

    @staticmethod
    async def _probe_once(client: httpx.AsyncClient, spec: CheckSpec) -> Tuple[bool, str]:
        try:
            resp = await client.get(spec.url, timeout=spec.timeout)
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"[:160]

        if spec.kind == "tts":
            # TTS 适配器有 /health，要求 JSON ok=true
            try:
                payload = resp.json()
            except Exception:  # noqa: BLE001
                return resp.status_code < 500, f"HTTP {resp.status_code}（无 JSON）"
            ok = bool(payload.get("ok")) if isinstance(payload, dict) else False
            return ok, "ok=true" if ok else f"HTTP {resp.status_code} ok={payload.get('ok') if isinstance(payload, dict) else '?'}"

        if spec.kind == "ollama":
            # /api/tags 能返回模型列表即视为存活
            if resp.status_code >= 500:
                return False, f"HTTP {resp.status_code}"
            return True, "HTTP %d" % resp.status_code

        # 通用：能连上并拿到任意 HTTP 响应就算活（whisper.cpp / GPT-SoVITS 对 GET / 会返回 404）
        return True, f"HTTP {resp.status_code}"

    async def _probe_with_retry(self, client: httpx.AsyncClient, spec: CheckSpec) -> Tuple[bool, str]:
        try:
            attempts = max(1, int(self.config.probe_attempts))
        except Exception:  # noqa: BLE001
            attempts = 3
        try:
            gap = max(0.0, float(self.config.probe_gap_seconds))
        except Exception:  # noqa: BLE001
            gap = 1.5

        last_detail = "未探测"
        for index in range(attempts):
            ok, detail = await self._probe_once(client, spec)
            if ok:
                return True, detail
            last_detail = detail
            if index < attempts - 1:
                await asyncio.sleep(gap)
        return False, last_detail

    async def _probe_all(self) -> Dict[str, Tuple[bool, str]]:
        specs = self._build_specs()
        results: Dict[str, Tuple[bool, str]] = {}
        if not specs:
            return results
        async with httpx.AsyncClient(follow_redirects=True, trust_env=False) as client:
            gathered = await asyncio.gather(
                *(self._probe_with_retry(client, spec) for spec in specs),
                return_exceptions=True,
            )
        for spec, outcome in zip(specs, gathered):
            if isinstance(outcome, BaseException):
                results[spec.key] = (False, f"{type(outcome).__name__}: {outcome}"[:160])
            else:
                results[spec.key] = outcome
        return results

    # ------------------------------------------------------------------ #
    # 告警
    # ------------------------------------------------------------------ #
    def _can_alert(self, key: str) -> bool:
        try:
            cooldown = max(0, int(self.config.cooldown_minutes)) * 60
        except Exception:  # noqa: BLE001
            cooldown = 3600
        return (time.time() - float(self._last_alert_at.get(key, 0.0))) >= cooldown

    async def _send_alert(self, text: str) -> bool:
        recipient = str(getattr(self.config, "alert_user_id", "") or "").strip()
        if not recipient:
            self.ctx.logger.warning("（未配置 alert_user_id）本应告警：%s", text.replace("\n", " / "))
            return False
        platform = str(getattr(self.config, "alert_platform", "qq") or "qq")
        try:
            stream = await self.ctx.chat.get_stream_by_user_id(recipient, platform)
        except Exception as exc:  # noqa: BLE001
            self.ctx.logger.warning("查私聊流失败 user=%s: %s", recipient, exc)
            return False
        stream_id = ""
        if isinstance(stream, dict):
            stream_id = str(stream.get("stream_id") or stream.get("session_id") or "")
        elif isinstance(stream, str):
            stream_id = stream
        if not stream_id:
            self.ctx.logger.warning("未找到 %s 的私聊流，告警未发送", recipient)
            return False
        try:
            ok = await self.ctx.send.text(text, stream_id, return_details=False)
        except Exception as exc:  # noqa: BLE001
            self.ctx.logger.warning("告警发送失败: %s", exc)
            return False
        self.ctx.logger.info("已发送告警（%s）: %s", ok, text.split("\n")[0])
        return bool(ok)

    async def _tick(self) -> None:
        results = await self._probe_all()
        specs = {spec.key: spec for spec in self._build_specs()}
        changed = False

        for key, (ok, detail) in results.items():
            previous = self._healthy.get(key)
            self._healthy[key] = ok
            self._last_detail[key] = detail
            changed = True
            if previous is True and not ok and self._can_alert(key):
                spec = specs.get(key)
                label = spec.label if spec else key
                hint = spec.hint if spec else ""
                message = (
                    f"【麦麦告警】{label} 没回应\n"
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"探测：{detail}\n"
                    f"{hint}\n"
                    f"（插件 local.health-watch；同类问题 {self.config.cooldown_minutes} 分钟内只提醒一次）"
                )
                if await self._send_alert(message):
                    self._last_alert_at[key] = time.time()

        if changed:
            self._save_state()

    # ------------------------------------------------------------------ #
    # 手动体检命令
    # ------------------------------------------------------------------ #
    @Command("health", pattern=r"^/(health|体检)\s*$")
    async def handle_health(self, **kwargs: Any) -> Any:
        """手动跑一次体检并把结果发在当前聊天里。"""
        stream_id = str(kwargs.get("stream_id") or "")
        results = await self._probe_all()
        specs = {spec.key: spec for spec in self._build_specs()}
        lines = ["【麦麦体检】" + time.strftime("%Y-%m-%d %H:%M:%S")]
        for key, (ok, detail) in results.items():
            spec = specs.get(key)
            label = spec.label if spec else key
            lines.append(f"{'✅' if ok else '❌'} {label}：{detail}")
        if not results:
            lines.append("（没有启用任何检查项）")
        report = "\n".join(lines)

        if stream_id:
            try:
                await self.ctx.send.text(report, stream_id)
            except Exception as exc:  # noqa: BLE001
                self.ctx.logger.warning("体检结果发送失败: %s", exc)
        # 空字符串：避免宿主再发一遍（内容已经自己发出去了）
        return True, "", 2

    # ------------------------------------------------------------------ #
    # 状态持久化
    # ------------------------------------------------------------------ #
    def _state_path(self) -> Optional[Path]:
        try:
            data_dir = Path(self.ctx.paths.data_dir)
        except Exception:  # noqa: BLE001
            return None
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001
            return None
        return data_dir / "state.json"

    def _save_state(self) -> None:
        path = self._state_path()
        if path is None:
            return
        payload = {
            "healthy": self._healthy,
            "last_alert_at": self._last_alert_at,
            "last_detail": self._last_detail,
            "discovered": sorted(self._discovered),
            "saved_at": time.time(),
        }
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            self.ctx.logger.warning("状态保存失败: %s", exc)

    async def _restore_state(self) -> None:
        path = self._state_path()
        if path is None or not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            self.ctx.logger.warning("状态读取失败: %s", exc)
            return
        if isinstance(payload, dict):
            self._healthy = {str(k): bool(v) for k, v in (payload.get("healthy") or {}).items()}
            self._last_alert_at = {str(k): float(v) for k, v in (payload.get("last_alert_at") or {}).items()}
            self._last_detail = {str(k): str(v) for k, v in (payload.get("last_detail") or {}).items()}
            self._discovered = {str(x) for x in (payload.get("discovered") or [])}


def create_plugin() -> HealthWatchPlugin:
    """插件入口。"""
    return HealthWatchPlugin()
