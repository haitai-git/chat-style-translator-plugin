"""聊天风格转译插件主入口。

功能：
1. 绑定 QQ 群后自动收集群友对话（支持消息屏蔽、字句黑名单），也可通过 NapCat
   历史消息接口回填（/风格拉取历史 或 WebUI「立即执行历史回填」），无需等待实时收集；
2. 达到设定条数或设定时间点后，在后台用 LLM 分析群聊场景、说话对象与群友个人风格，
   生成「风格模型」并经过字句黑名单与语义黑名单过滤；
3. 支持多模型融合（融合时同样执行黑名单过滤）、模型内语句/场景删除、黑名单管理；
4. 审核通过并启用后，AI 的自身行为风格保持不变；每次 AI 生成回复草稿后，
   检索该群风格模型并让 LLM 按可调强度对草稿做微调，原意严格不变；
5. 分析进度与模型融合进度会私聊发送给配置的 QQ 号。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from maibot_sdk import Command, Field, HookHandler, MaiBotPlugin, PluginConfigBase
from maibot_sdk.types import ErrorPolicy, HookMode, HookOrder

from .modules.analyzer import StyleAnalyzer
from .modules.garbage_guard import detect_garbage
from .modules.injector import build_rewrite_reference
from .modules.storage import StyleStore

logger = logging.getLogger("chat_style_translator")

# 会话-群映射的最大条目数；超过时淘汰最旧的条目，防止长期运行内存无限增长
_MAX_SESSION_MAP_ENTRIES = 2000

# 风格预设 → LLM 提示词要求（用于微调前黑名单的「删词补意风格」与回复后微调的「微调风格」）
# 每个预设同时约束「语气」与「用词」（是否允许网络梗词/流行语），避免出现风格错配的用词
REWRITE_STYLE_GUIDE: Dict[str, str] = {
    "无": "",
    "温柔": "输出整句话的语气与措辞要温柔、友善、语气软和，用词温和，避免网络梗词，避免生硬或攻击性的用词",
    "自然": "输出整句话要口语化、自然，像真人日常聊天一样，不生硬、不像机器人腔，可用少量网络梗但要自然不突兀",
    "活泼": "输出整句话要轻快、有活力，可以适当使用语气词让句子更生动，可以适当玩梗",
    "直爽": "输出整句话要直接利落、干脆不绕弯，像说话直爽的人一样，不扭捏，少用网络梗词",
    "软萌": "输出整句话要软糯可爱、亲切，像软萌性格的人说话一样，避免粗俗或生硬的梗词",
    "沉稳": "输出整句话要成熟稳重、有条理，像沉稳的人说话一样，不毛躁，用词正式克制，禁止网络梗词",
    "冷漠": "输出整句话要语气冷淡、简短克制、有距离感，用词简洁平淡，禁止网络梗词，但不得删改内容或改变原意",
    "热情": "输出整句话要语气热情饱满、积极亲近、有感染力，用词可以生动一点",
}

# 风格预设 → WebUI 下拉选项注释（显示在「删词补意风格」与「微调风格」字段下方提示里）
REWRITE_STYLE_NOTES: Dict[str, str] = {
    "无": "不加任何风格（默认）",
    "温柔": "语气温柔友善、软和；用词温和，避免网络梗词",
    "自然": "口语自然，像正常人聊天；可少量用梗",
    "活泼": "轻快有活力，可带语气词，可适当玩梗",
    "直爽": "直接利落、干脆不绕弯；少用网络梗词",
    "软萌": "软糯可爱、亲切；避免粗俗或生硬的梗词",
    "沉稳": "成熟稳重、有条理；用词正式克制，禁止网络梗词",
    "冷漠": "冷淡克制、有距离感；用词简洁平淡，禁止网络梗词",
    "热情": "热情饱满、积极有感染力",
    "自定义": "使用下方「自定义风格描述」的文本",
}

# 「不用符号表达情绪」：开启时追加到微调与删词补意提示词里的硬性要求，
# 防止 LLM 用波浪号/省略号等符号来"配音"营造语气，显得不像真人聊天
NO_SYMBOL_EMOTION_RULE = (
    "禁止使用波浪号、省略号（如 ~、～、…… 等）等符号来承担语气或表达情绪，"
    "情绪一律用中文文字自然表达，像真人聊天一样，不要为了贴合风格硬加任何装饰符号"
)

HELP_TEXT = """聊天风格转译插件使用说明

【工作原理】
插件不会改变 AI 自身的行为风格，也不会往 AI 的提示词里注入风格指令；
每次 AI 生成回复草稿后，插件会检索当前群的风格模型，让 LLM 对草稿做微调——
微调强度可在 WebUI 插件配置 → 回复后微调 → 微调强度（滑块 0~100）中调整：
0 = 几乎不改，30 = 轻度（默认，只改几个词），50 = 中等，80+ = 明显贴合群聊风格；
任何强度下原意、事实、数字、人名、链接都严格不变；微调失败或超时则原样发送。
还可以在 回复后微调 → 微调风格 中选择语气风格（与「删词补意风格」相同选项，默认「无」），
微调会优先按该风格调整措辞与语气（与群聊风格冲突时以风格设置为准，原意不变）；
注意：选了风格（非「无」）后，会按所选风格微调、跳过按群模型风格的微调（群风格不生效），
选「无」才完全按群模型风格微调。
微调模型任务名强烈建议填写小而快的模型（如 utils、deepseek-v4-flash），减少回复耗时。

【第一步：绑定群并收集对话】
方式一（WebUI 配置）：插件配置 → 群绑定与模型 → 添加 QQ 群号，并填写该群使用的模型（可留空）
方式二（群内命令）：
/风格绑定群 <群号>        绑定群，开始自动收集群友对话（<群号> 必填）
/风格解绑群 <群号>        解绑群并删除收集的对话（<群号> 必填）
/风格状态                查看各群收集进度与自动任务状态（无需参数）
/风格生成 [群号]          后台转译分析（立即回复；进度与结果私聊通知）
/风格拉取历史 [群号] [条数]  后台拉取 NapCat 历史消息回填（立即回复；结果私聊通知）
（[群号] 表示可选：群内执行 = 当前群；私聊执行且不指定群号 = 全部绑定群；
  进度/结果通知发送给「进度私聊 QQ 号」填写的 QQ，未配置则静默不发送）
（从配置里删除的群会自动解绑；命令绑定的群不受配置同步影响）

【历史消息回填（不用等实时收集）】
也可以：插件配置 → 收集与自动分析 → 历史回填条数（默认 300）+「立即执行历史回填」开关，
保存配置后自动对全部绑定群回填（回填完成后请把开关关回）。
回填与实时收集执行相同的过滤：用户/机器人屏蔽、带 / 的句子、字句黑名单、跳过本人消息，
并按消息 ID 去重；回填后收集条数达到自动分析阈值会自动触发分析。
需要机器人在群里、NapCat 在线且该群允许获取历史记录，否则拉取不到。

【第二步：审核模型】
（WebUI 插件配置 → 插件设置 →「模型目录自动生效」默认开启：把模型文件放进模型目录就
  会自动审核通过并对所有群生效，无需任何命令；关闭后可回到下面的 /风格启用 手动挂载流程）
（<模型名> 必填，可用文件名、模型名或模型 ID 引用，例如 /风格启用 1 或 /风格启用 群1 风格模型）
/风格列表                查看全部风格模型（无需参数）
/风格详情 <模型名>        查看模型摘要
/风格场景 <模型名> [序号]  查看模型某个场景的语句明细（序号不填 = 查看全部场景）
/风格启用 <模型名> [群号]   审核通过并挂载到群，AI 回复时生效
/风格停用 <模型名> [群号]   从群上摘除模型
/风格全局启用 <模型名>     开启全局通用：该模型对所有群聊都生效
/风格全局停用 <模型名>     关闭全局通用
（/风格启用 与 /风格停用 的 [群号] 不填时：群内执行 = 当前群；私聊执行 = 必须填写群号，
  否则会提示「请在群内执行，或指定群号」）
（群聊通用开关也可以在 WebUI 插件配置 → 回复后微调 → 群聊通用模型 ID 中直接编辑；
  开启群聊通用后，AI 回复直接使用通用模型，忽略各群单独配置的模型）
/风格删除语句 <模型名> <语句ID>     （两个参数都必填）
/风格删除场景 <模型名> <场景ID或序号>
/风格重命名 <模型名> <新名>
/风格删除模型 <模型名>

【多模型融合】
/风格合并 <模型A> <模型B> [新名]    （模型A、模型B 必填；新名不填自动生成）
相似场景与相似含义的语句会自动融合去重，可多次合并出更完整的模型。
也可以在 WebUI 插件配置 → 模型融合 中填写模型 A/B 并打开「执行融合」，
保存配置后自动后台融合（开关保持打开时，每次保存配置或插件重载都会重新融合
并生成新模型，不覆盖原模型；融合完成后建议把开关关回）。
融合优化（WebUI 模型融合节）：融合比例（滑块 A:B）——含义相同的重复语句
只保留一条，并按比例优先保留某一方的表述；去重强度——标准 / 激进精简（相近
含义也合并，体积更小）/ 不去重；语句总数上限——限制融合后总语句数，防止体积过大；
场景合并阈值与风格快照合并阈值——控制相似内容自动合并的力度（滑块，跨度大的内容会错开保留）。
融合结果会生成新的模型文件（不会覆盖原模型），并执行字句与语义黑名单过滤。
融合进度（0% 开始、30% 融合中、70% 过滤中、100% 完成）会私聊发送给
「进度私聊 QQ 号」填写的 QQ；失败与跳过也会通知。

【单模型精简（模型太大时压缩体积）】
WebUI 插件配置 → 模型精简：填写「要精简的模型」（模型名或 ID，/风格列表 可查看），
可选填「新模型名」（留空自动生成「原名（精简版）」），打开「执行精简」开关并保存配置，
插件会后台自动把该模型内相似场景、近似语句、相似风格快照合并去重（纯本地处理，
不调用 LLM、不改变语句内容，秒出结果），并执行字句与语义黑名单过滤。
精简力度可调：去重强度（标准 / 激进精简 / 不去重）、语句总数上限（超限时优先保留
更长更完整的语句）、场景合并阈值与风格快照合并阈值（与融合共用同一套滑块逻辑）。
精简结果会生成全新的模型文件（原模型不动），状态为待审核，用 /风格列表 查看、
/风格启用 挂载即可；开关保持打开时，每次保存配置或插件重载都会重新精简并生成
新模型（原模型不动），精简完成后建议把开关关回。进度与结果私聊发送给「进度私聊 QQ 号」。

【模型文件】
模型以 JSON 文件保存在模型目录（config.toml 的 storage.model_dir，
默认 data/plugins/local.chat-style-translator/models/），可手动打开编辑、
重命名或增删文件，改动会立即生效。

【黑名单】
/黑名单添加 <语句>        通用拉黑：按语义相似度对所有模型过滤该语句的意思
/黑名单添加模型 <模型名> <语句>  只在该模型内拉黑该语句的意思
/黑名单删除 <ID>
/黑名单列表
/黑名单清空
（黑名单也可以在 WebUI 插件配置 → 黑名单 中编辑：可填含义描述、限定模型、开启通用拉黑）

【字句黑名单】
在 WebUI 插件配置 → 字句黑名单 中填写字句（一条一行）：
收集时消息文本完整包含任一字句就会被跳过（整句子串匹配，例如「修仙」不会误伤只含「修」的消息）。

【微调前黑名单】（WebUI 插件配置 → 微调前黑名单，默认关闭）
AI 生成回复草稿后、风格微调之前，先把草稿中出现的黑名单词删掉，再进行微调：
仅支持中文词条（词条必须含中文才生效，纯英文、数字、符号与 emoji 词条无效）；
字面匹配为整词/子串匹配——例如填写「涩涩」，草稿中出现「涩涩」就会被删除，只含单个字的不会误伤；
开启「语义换词匹配」后，删除会基于「没删前的整句话」理解意思后进行，并尽量保持原有意思：
删除造成句子不通顺或丢失语气时允许补字词或换近义表达（例如删「属实」可补「是」，
删「绷不住了」可写「难绷」），只要字面不出现黑名单词即可；
同义换词、变形、谐音等表述也会被一并删除；语义换词匹配生效后不再进行回复后风格微调，
只返回删词补意后的文本（删词补意本身已保持原意并贴合「删词补意风格」）；仅字面删词命中时仍会继续回复后微调。
「删词补意风格」（下拉选择，默认「无」）控制删词后整句话的语气与措辞倾向：
无 = 什么都不额外加；温柔 / 自然（像正常人聊天）/ 活泼 / 直爽 / 软萌 / 沉稳 / 冷漠 / 热情 为预设性格风格；
选「自定义」时用「自定义风格描述」输入框自由描述想要的语气（如"像东北老铁唠嗑"），留空则不注入。
只影响语义换词匹配环节的输出，让删词补意后的句子更贴合所选风格。
开启后每条草稿会额外消耗一次模型调用；删除后草稿为空时放弃微调并保留原回复；
微调失败或超时时仍返回删词后的文本，保证黑名单词不会出现在最终回复里。
只作用于微调流程，不影响收集与模型分析。

【消息屏蔽】（WebUI 插件配置 → 消息屏蔽）
跳过收集本人消息（默认开）：机器人自己的消息不收集进模型，自动识别机器人账号，
识别不到时在「本人 QQ 号」中手动填写；
屏蔽用户 QQ 号：这些用户的群消息不会被收集；
自动屏蔽带 / 的句子；自动屏蔽 QQ 机器人和官方机器人。

管理命令仅管理员（config.toml 的 owner.qq）或 WebUI 本地操作者可用。
"""


class PluginSection(PluginConfigBase):
    """插件基础设置。"""

    __ui_label__ = "插件设置"
    __ui_icon__ = "settings"
    __ui_order__ = 90

    enabled: bool = Field(
        default=True,
        description="是否启用插件",
        json_schema_extra={"label": "启用插件"},
    )
    no_symbol_emotion: bool = Field(
        default=True,
        description="微调/删词补意时禁止用波浪号、省略号等符号表达语气情绪（如 ~、～、……），情绪一律用中文文字自然表达，避免显得不像真人",
        json_schema_extra={
            "label": "不用符号表达情绪",
            "hint": "默认开启：风格微调与删词补意时不使用 ~、～、…… 等符号来营造语气或情绪，情绪用文字自然表达；关闭后可放行符号",
        },
    )
    auto_enable_models: bool = Field(
        default=True,
        description="模型目录自动生效：把模型文件放进模型目录后，自动审核通过并对所有群生效，无需再用 /风格启用 挂载",
        json_schema_extra={
            "label": "模型目录自动生效",
            "hint": "默认开启：模型目录里每个模型文件都会自动审核通过并对所有绑定群生效（等同全局通用），放进去即生效、无需命令；关闭后回到原有流程，需用 /风格启用 或配置 global_models 手动挂载",
        },
    )
    config_version: str = Field(
        default="0.18.0",
        description="配置版本号，升级配置结构时递增",
        json_schema_extra={"label": "配置版本号", "disabled": True},
    )


class OwnerSection(PluginConfigBase):
    """管理员 QQ 号列表，管理命令（绑定群、生成、黑名单等）只有管理员可用。"""

    __ui_label__ = "管理员"
    __ui_icon__ = "user-cog"
    __ui_order__ = 70

    qq: List[str] = Field(
        default_factory=list,
        description="管理员 QQ 号列表，多个用逗号分隔；留空时管理命令全部拒绝",
        json_schema_extra={"label": "管理员 QQ 号", "placeholder": "例如 123456789"},
    )


class StorageSection(PluginConfigBase):
    """模型文件保存位置。模型以 JSON 文件保存，可直接手动编辑、重命名或增删。"""

    __ui_label__ = "模型存储"
    __ui_icon__ = "folder"
    __ui_order__ = 80

    model_dir: str = Field(
        default="",
        description="模型文件保存目录；留空使用插件数据目录下的 models 文件夹，支持绝对路径或相对路径",
        json_schema_extra={
            "label": "模型保存目录",
            "placeholder": "留空使用默认目录；例如 D:\\StyleModels",
            "hint": "修改后需重载插件生效；更换目录不会自动迁移旧模型",
        },
    )


class GroupBindingItem(PluginConfigBase):
    """单个群的绑定与模型配置。"""

    __ui_label__ = "群绑定"

    group_id: str = Field(
        default="",
        description="要收集消息的 QQ 群号",
        json_schema_extra={"label": "QQ 群号", "placeholder": "例如 123456789"},
    )
    platform: str = Field(
        default="qq",
        description="平台",
        json_schema_extra={"label": "平台", "disabled": True, "hidden": True},
    )
    models: List[str] = Field(
        default_factory=list,
        description="该群使用的模型（模型名或 ID，多个用逗号分隔）；开启群聊通用后忽略此配置",
        json_schema_extra={
            "label": "使用模型",
            "hint": "填写模型名或模型 ID（/风格列表 可查看）；留空表示只收集不注入；群聊通用开启时直接用通用模型",
        },
    )


class GroupsSection(PluginConfigBase):
    """在这里添加要收集消息的 QQ 群，并指定每个群使用的风格模型。从列表删除的群会自动停止收集。"""

    __ui_label__ = "群绑定与模型"
    __ui_icon__ = "users"
    __ui_order__ = 10

    items: List[GroupBindingItem] = Field(
        default_factory=list,
        description="要收集消息的群列表，以及每个群使用的模型",
        json_schema_extra={"label": "群列表"},
    )


class MergeSection(PluginConfigBase):
    """配置触发式自动融合：填好模型 A 与模型 B，打开「执行融合」开关并保存配置，插件会自动融合生成新模型。

    融合优化：融合比例（A:B）决定含义相同/相似的重复语句优先保留哪一方的表述；去重强度控制语句精简程度；
    语句总数上限可限制融合后模型体积。融合结果为新模型（状态 pending），可用 /风格列表 查看、/风格启用 挂载；
    插件无法回写配置开关：开关保持打开时，每次保存配置或插件重载都会重新融合生成新模型（不覆盖原模型），
    融合完成后建议把开关关回。
    """

    __ui_label__ = "模型融合"
    __ui_icon__ = "git-branch"
    __ui_order__ = 15

    model_a: str = Field(
        default="",
        description="要融合的模型 A（模型名或 ID）",
        json_schema_extra={"label": "模型A", "placeholder": "模型名或 ID，例如 群1 风格模型"},
    )
    model_b: str = Field(
        default="",
        description="要融合的模型 B（模型名或 ID）",
        json_schema_extra={"label": "模型B", "placeholder": "模型名或 ID，例如 群2 风格模型"},
    )
    new_name: str = Field(
        default="",
        description="新模型名，留空自动生成",
        json_schema_extra={"label": "新模型名（可选）", "placeholder": "留空自动命名"},
    )
    trigger: bool = Field(
        default=False,
        description="执行融合：打开并保存配置后自动开始融合；融合完成后请将开关关回",
        json_schema_extra={
            "label": "执行融合",
            "hint": "保存配置后自动开始融合（后台执行）。开关保持打开时，每次保存配置或插件重载都会重新融合并生成新模型（不覆盖原模型）；融合完成后请将开关关回",
            "order": 0,
        },
    )
    ratio_a: int = Field(
        default=50,
        ge=0,
        le=100,
        description="融合比例（模型A占比，0~100）：两模型出现含义相同/高度相似的重复语句时，只保留一条，并按 A:B 比例决定优先保留哪一方的表述，避免重复堆积导致模型体积过大",
        json_schema_extra={
            "label": "融合比例（A:B）",
            "x-widget": "slider",
            "step": 5,
            "hint": "滑块调到 70 表示 A:B = 7:3：含义相同的重复语句约 70% 保留 A 方表述、30% 保留 B 方；50 表示五五开；仅对两个模型融合时生效",
            "order": 1,
        },
    )
    dedup_level: Literal["标准", "激进精简", "不去重"] = Field(
        default="标准",
        description="融合去重强度：标准=完全重复或含义高度相似的语句只保留一条；激进精简=含义相近的语句也合并、相近场景也合并，进一步压缩体积；不去重=两边语句都保留",
        json_schema_extra={
            "label": "去重强度",
            "hint": "标准：完全重复/高度相似只留一条；激进精简：相近含义也合并（体积更小）；不去重：两边都保留（体积更大）",
            "order": 2,
        },
    )
    max_total_statements: int = Field(
        default=0,
        ge=0,
        description="融合后模型语句总数上限（所有场景合计，0 表示不限制）：防止模型体积过大",
        json_schema_extra={
            "label": "语句总数上限",
            "hint": "0 = 不限制；例如填 120，融合后所有场景合计最多保留 120 条最有代表性的语句",
            "order": 3,
        },
    )
    scene_merge_threshold: float = Field(
        default=0.8,
        ge=0.5,
        le=0.95,
        description="场景合并相似度阈值（0.5~0.95）：融合时内容相似或重复的场景自动合并为一个（场景名相同/互相包含或语句大面积重叠的直接合并，与来源、时段无关；其余按该阈值判定），方便 AI 直接按场景检索",
        json_schema_extra={
            "label": "场景合并阈值",
            "x-widget": "slider",
            "step": 0.01,
            "hint": "默认 0.8：只合并高度相似的场景（如「监狱保释梗」与「监狱/保释梗」）；调低 = 更多相近场景合并（模型更紧凑、检索更直接），调高 = 更多场景各自独立保留",
            "order": 4,
        },
    )
    snapshot_merge_threshold: float = Field(
        default=0.55,
        ge=0.3,
        le=0.85,
        description="整体风格快照合并相似度阈值（0.3~0.85）：融合时相似的风格合并为一条，差异大的风格错开保留",
        json_schema_extra={
            "label": "风格快照合并阈值",
            "x-widget": "slider",
            "step": 0.01,
            "hint": "默认 0.55：相似的风格合并为一条（保留更长更完整的表述）；调低 = 快照合并更多（数量更少），调高 = 快照更分散（各自独立保留）",
            "order": 5,
        },
    )


class CompactSection(PluginConfigBase):
    """单模型精简：模型太大时，把模型内相似场景、近似语句、相似风格快照自动合并去重。
    相似或重复的场景必定合并（场景名相同/互相包含或语句大面积重复时直接合并，与其原来源/时段无关）；

    纯本地处理、不调用 LLM、不改变语句内容，只减少重复/相似内容，秒出结果；
    精简结果会生成全新模型文件（原模型不动），审核后 /风格启用 挂载即可。
    """

    __ui_label__ = "模型精简"
    __ui_icon__ = "scissors"
    __ui_order__ = 16

    model: str = Field(
        default="",
        description="要精简的模型（模型名或 ID）",
        json_schema_extra={"label": "要精简的模型", "placeholder": "模型名或 ID，例如 群1 风格模型"},
    )
    new_name: str = Field(
        default="",
        description="精简后新模型名，留空自动生成（原名 + 「精简版」）",
        json_schema_extra={"label": "新模型名（可选）", "placeholder": "留空自动命名"},
    )
    trigger: bool = Field(
        default=False,
        description="执行精简：打开并保存配置后自动开始；精简完成后请将开关关回",
        json_schema_extra={
            "label": "执行精简",
            "hint": "保存配置后自动开始精简（后台执行）。开关保持打开时，每次保存配置或插件重载都会重新精简并生成新模型（原模型不动）；精简完成后请将开关关回",
            "order": 0,
        },
    )
    dedup_level: Literal["标准", "激进精简", "不去重"] = Field(
        default="标准",
        description="精简去重强度：标准=完全重复的语句只保留一条；激进精简=含义相近的语句也合并、相近场景也合并，压缩最明显；不去重=只合并相似场景、语句全部保留",
        json_schema_extra={
            "label": "去重强度",
            "hint": "标准：完全重复只留一条；激进精简：近似句也合并（体积更小）；不去重：只合并相似场景",
            "order": 1,
        },
    )
    max_total_statements: int = Field(
        default=0,
        ge=0,
        description="精简后模型语句总数上限（所有场景合计，0 表示不限制）：超限时优先保留更长更完整的语句",
        json_schema_extra={
            "label": "语句总数上限",
            "hint": "0 = 不限制；例如填 120，精简后所有场景合计最多保留 120 条最有代表性的语句",
            "order": 2,
        },
    )
    scene_merge_threshold: float = Field(
        default=0.8,
        ge=0.5,
        le=0.95,
        description="场景合并相似度阈值（0.5~0.95）：精简时内容相似或重复的场景自动合并为一个（场景名相同/互相包含或语句大面积重叠的直接合并，与其原来源/时段无关；其余按该阈值判定）",
        json_schema_extra={
            "label": "场景合并阈值",
            "x-widget": "slider",
            "step": 0.01,
            "hint": "默认 0.8：只合并高度相似的场景；调低 = 更多相近场景合并（模型更紧凑），调高 = 场景更分散",
            "order": 3,
        },
    )
    snapshot_merge_threshold: float = Field(
        default=0.55,
        ge=0.3,
        le=0.85,
        description="整体风格快照合并相似度阈值（0.3~0.85）：精简时相似的风格合并为一条，差异大的风格错开保留",
        json_schema_extra={
            "label": "风格快照合并阈值",
            "x-widget": "slider",
            "step": 0.01,
            "hint": "默认 0.55：相似的风格合并为一条（保留更长更完整的表述）；调低 = 快照合并更多，调高 = 快照更分散",
            "order": 4,
        },
    )


class UserBlockSection(PluginConfigBase):
    """屏蔽不想收集的消息：用户黑名单、带 / 的句子、QQ 机器人与官方机器人、机器人本人。"""

    __ui_label__ = "消息屏蔽"
    __ui_icon__ = "shield-ban"
    __ui_order__ = 60

    qq: List[str] = Field(
        default_factory=list,
        description="屏蔽用户 QQ 号列表：这些用户的群消息不会被收集进模型",
        json_schema_extra={
            "label": "屏蔽用户 QQ 号",
            "hint": "输入 QQ 号即可屏蔽该用户，多个用逗号分隔；被屏蔽用户的消息不会进入模型学习",
            "order": 0,
        },
    )
    skip_self_messages: bool = Field(
        default=True,
        description="跳过收集本人聊天：机器人自己发的消息不会收集进模型（自动识别机器人账号）",
        json_schema_extra={
            "label": "跳过收集本人消息",
            "hint": "默认开启：机器人自己的消息不会被收集进风格模型。会自动识别机器人账号（NapCat 适配器默认已过滤本人消息），识别不到时请在下方手动填写本人 QQ 号",
            "order": 1,
        },
    )
    self_qq: List[str] = Field(
        default_factory=list,
        description="本人 QQ 号列表（手动补充，自动识别不生效时填写）",
        json_schema_extra={
            "label": "本人 QQ 号",
            "hint": "自动识别不生效时手动填写机器人自己的 QQ 号，多个用逗号分隔",
            "order": 2,
        },
    )
    block_slash: bool = Field(
        default=True,
        description="自动屏蔽带 / 的句子（类似命令的消息不收集）",
        json_schema_extra={"label": "屏蔽带 / 的句子", "order": 3},
    )
    block_qq_bots: bool = Field(
        default=True,
        description="自动屏蔽 QQ 机器人和官方机器人（内置识别官方机器人 QQ 段）",
        json_schema_extra={
            "label": "屏蔽机器人与官方机器人",
            "hint": "自动识别 QQ 官方机器人 QQ 段（285419631 开头），可配合下方机器人 QQ 号补充",
            "order": 4,
        },
    )
    bot_ids: List[str] = Field(
        default_factory=list,
        description="额外的机器人 QQ 号列表（手动补充）",
        json_schema_extra={"label": "机器人 QQ 号", "hint": "除了内置的官方机器人段外，额外屏蔽这些机器人 QQ 号", "order": 5},
    )


class CollectionSection(PluginConfigBase):
    """控制收集多少对话后自动分析，以及每日自动分析时间。"""

    __ui_label__ = "收集与自动分析"
    __ui_icon__ = "message-circle"
    __ui_order__ = 30

    auto_analyze_count: int = Field(
        default=100,
        description="收集满多少条消息后自动后台分析（0 关闭按条数触发）",
        json_schema_extra={"label": "自动分析条数", "hint": "0 = 不按条数自动分析，只用手动 /风格生成"},
    )
    daily_time: str = Field(
        default="",
        description="每日自动分析时间点（HH:MM），留空关闭",
        json_schema_extra={"label": "每日分析时间", "placeholder": "例如 21:00"},
    )
    max_messages: int = Field(
        default=300,
        description="单次分析最多使用的消息条数",
        json_schema_extra={"label": "单次分析消息上限", "hint": "只取最近 N 条，防止分析内容过长"},
    )
    max_stored_messages: int = Field(
        default=500,
        description="每个群最多存储的消息条数，超出后自动删除最旧消息（0 关闭清理）",
        json_schema_extra={
            "label": "每群存储上限",
            "hint": "防止消息文件无限增长占用硬盘：超过上限只保留最近 N 条（建议 ≥ 单次分析消息上限）；0 = 不自动清理",
            "order": 3,
        },
    )
    window_size: int = Field(
        default=60,
        description="每次 LLM 分析的消息窗口条数",
        json_schema_extra={"label": "分析窗口条数", "hint": "消息过多时分批分析再融合，一般保持默认"},
    )
    backfill_count: int = Field(
        default=300,
        ge=0,
        description="历史回填条数：从 NapCat 群聊历史中拉取该群最近多少条消息补进收集池（0 关闭）",
        json_schema_extra={
            "label": "历史回填条数",
            "hint": "用 /风格拉取历史 或「立即执行历史回填」时，拉取该群最近 N 条历史消息（0 = 不拉取）；拉取与实时收集一致执行屏蔽与黑名单过滤、按消息 ID 去重",
            "order": 4,
        },
    )
    backfill_batch_size: int = Field(
        default=50,
        ge=1,
        le=100,
        description="每次请求 NapCat 拉取的历史消息条数（单次上限 100）",
        json_schema_extra={
            "label": "回填单批条数",
            "hint": "NapCat 单次请求建议不超过 100 条；条数多分页次数少，但响应更慢",
            "order": 5,
        },
    )
    trigger_analyze_now: bool = Field(
        default=False,
        description="打开并保存配置后，立即对全部已绑定群执行一次后台分析（分析完成后请将开关关回）",
        json_schema_extra={"label": "立即执行分析", "hint": "开关打开 → 保存配置 → 后台立即分析全部绑定群；完成后关回开关即可", "order": 99},
    )
    trigger_backfill_now: bool = Field(
        default=False,
        description="打开并保存配置后，立即对全部已绑定群执行一次历史消息回填（回填完成后请将开关关回）",
        json_schema_extra={
            "label": "立即执行历史回填",
            "hint": "开关打开 → 保存配置 → 后台立即对全部绑定群拉取历史消息补进收集池；完成后关回开关即可",
            "order": 98,
        },
    )


class AnalysisSection(PluginConfigBase):
    """转译分析使用的模型与生成参数，一般保持默认即可。"""

    __ui_label__ = "分析参数"
    __ui_icon__ = "cpu"
    __ui_order__ = 40

    model: str = Field(
        default="",
        description="分析使用的模型任务名，留空使用默认模型",
        json_schema_extra={"label": "分析模型任务名", "placeholder": "留空使用默认模型"},
    )
    notify_qq: str = Field(
        default="",
        description="分析进度/结果私聊发送的 QQ 号（需是好友）；留空则不发送任何进度通知",
        json_schema_extra={
            "label": "进度私聊 QQ 号",
            "placeholder": "留空 = 不发送进度通知",
            "hint": "填写后，自动/按钮触发的分析进度与结果会私聊发送给该 QQ；不是好友时静默不发送",
            "order": 2,
        },
    )
    max_statements_per_scene: int = Field(
        default=30,
        description="融合后每个场景最多保留的语句数",
        json_schema_extra={"label": "每场景最大语句数"},
    )
    analyze_concurrency: int = Field(
        default=2,
        description="并发分析的批次数（1~4）。并发不改变分析内容，只加快速度；过高可能触发模型限流",
        json_schema_extra={"label": "并发分析批数", "hint": "2 = 同时分析 2 批对话窗口，建议 1~3", "step": 1},
    )
    progress_interval_seconds: float = Field(
        default=120.0,
        description="进度通知发送间隔（秒）：按「已处理 X/总 Y 条（Z%）」总进度发送；0 = 每批完成都发送",
        json_schema_extra={"label": "进度通知间隔（秒）", "hint": "默认 120 = 约每 2 分钟发一次总进度", "step": 10},
    )


class BlacklistItem(PluginConfigBase):
    """单条黑名单配置。"""

    __ui_label__ = "黑名单条目"

    text: str = Field(
        default="",
        description="要拉黑的语句（原话或意思描述）",
        json_schema_extra={"label": "拉黑语句", "placeholder": "例如：骂人的话"},
    )
    meaning: str = Field(
        default="",
        description="语句含义描述（可选，用于语义相似度匹配）",
        json_schema_extra={"label": "含义描述", "placeholder": "例如：表达强烈不满和辱骂"},
    )
    model: str = Field(
        default="",
        description="限定模型（模型名或 ID，留空表示所有模型）",
        json_schema_extra={"label": "限定模型", "placeholder": "留空表示所有模型"},
    )
    apply_all: bool = Field(
        default=True,
        description="通用拉黑开关：开启后按语义相似度对所有模型生效",
        json_schema_extra={"label": "通用拉黑", "hint": "开启 = 所有模型都按相似度拉黑；关闭时仅对上面限定的模型生效"},
    )


class BlacklistSection(PluginConfigBase):
    """拉黑语句的意思：含义高度相似的内容不会进入模型，可限定模型或全局生效。"""

    __ui_label__ = "黑名单"
    __ui_icon__ = "shield-x"
    __ui_order__ = 50

    similarity_threshold: float = Field(
        default=0.85,
        description="语义相似度阈值（0~1），高于则过滤",
        json_schema_extra={
            "label": "相似度阈值",
            "step": 0.01,
            "hint": "越高越严格：0.5 宽松，0.9 只过滤含义几乎一致的语句",
        },
    )
    items: List[BlacklistItem] = Field(
        default_factory=list,
        description="要拉黑的语句意思列表（可限定模型，可开启通用拉黑）",
        json_schema_extra={"label": "黑名单列表"},
    )


class PhraseBlockSection(PluginConfigBase):
    """字句黑名单：消息文本完整包含其中任一字句时，该消息不会被收集进模型。

    整句子串匹配：例如填写「修仙」，只有消息里完整出现「修仙」二字才会被跳过，
    只含单个「修」字的消息不受影响。
    """

    __ui_label__ = "字句黑名单"
    __ui_icon__ = "filter"
    __ui_order__ = 55

    items: List[str] = Field(
        default_factory=list,
        description="黑名单字句列表：一条一行；消息文本完整包含任一字句即跳过不收集",
        json_schema_extra={
            "label": "字句列表",
            "hint": "一条一行；例如填写「修仙」，消息里完整包含「修仙」二字才会被跳过（单字不算）；已收集的消息在下次分析时也会被过滤",
        },
    )


class InjectSection(PluginConfigBase):
    """控制 AI 回复后如何按群聊风格做轻度微调，包括群聊通用与群友风格匹配。

    插件不会把风格模型注入 AI 的提示词，因此 AI 聊天时的自身行为风格保持不变；
    只在 AI 生成回复草稿后，检索风格并让 LLM 做轻度微调（只改几个词，原意不变）。
    """

    __ui_label__ = "回复后微调"
    __ui_icon__ = "zap"
    __ui_order__ = 20

    enabled: bool = Field(
        default=True,
        description="是否在 AI 生成回复后按群聊风格做轻度微调（只改几个词，原意不变；AI 自身行为风格不受影响）",
        json_schema_extra={"label": "启用微调", "order": 0},
    )
    enable_style_matching: bool = Field(
        default=True,
        description="按当前说话人风格微调：优先参考当前说话人（QQ号）的个人聊天风格做轻度微调；关闭后仅参考群整体风格",
        json_schema_extra={
            "label": "按说话人风格微调",
            "hint": "开启后优先参考当前说话人的个人风格；关闭后只参考群整体风格",
            "order": 1,
        },
    )
    global_models: List[str] = Field(
        default_factory=list,
        description="群聊通用模型 ID 列表：列表中的模型对所有群聊（含未绑定群）生效",
        json_schema_extra={
            "label": "群聊通用模型 ID",
            "hint": "填写模型 ID（用 /风格列表 查看，形如 m_2026…），多个用逗号分隔；开启后所有群直接用通用模型，忽略各群单独配置的模型",
            "order": 2,
        },
    )
    max_chars: int = Field(
        default=2500,
        description="单次微调使用的风格参考文本最大字符数",
        json_schema_extra={
            "label": "参考文本上限",
            "hint": "参考文本超长时按优先级取舍：整体风格、说话人档案、群友档案最先保留，场景与语句按顺序塞入，超出上限的部分丢弃，最后再硬截断到上限。调大（如 5000）可保留更多场景语句，但会挤占微调请求空间、变慢",
            "order": 3,
        },
    )
    rewrite_model: str = Field(
        default="",
        description="风格微调使用的模型任务名，留空使用默认模型；强烈建议使用小而快的模型来减少回复时间",
        json_schema_extra={
            "label": "微调模型任务名",
            "placeholder": "留空使用默认模型",
            "hint": "强烈建议填写小而快的模型（如 utils、deepseek-v4-flash）：风格微调只是轻度改写，小模型足够且响应更快，能明显减少每次回复的额外耗时",
            "order": 4,
        },
    )
    rewrite_strength: int = Field(
        default=30,
        ge=0,
        le=100,
        description="微调强度（0~100）：越高改写得越明显（用词、语气、句式贴合群聊风格），0 表示几乎不改；无论多高都保证原意严格不变",
        json_schema_extra={
            "label": "微调强度",
            "x-widget": "slider",
            "step": 5,
            "hint": (
                "0 = 完全不动，直接发原回复；"
                "1~24 = 几乎不动，只调语气词/标点；"
                "25~44 = 轻度（默认 30）：只改几个词，句式基本不变；"
                "45~64 = 中等：换用词、调语气、微调句式，更贴群聊味；"
                "65~84 = 较强：明显换词改句，接近群友说话方式；"
                "85~100 = 强烈：大幅调整措辞语气，充分贴合群风格（仍保原意）。"
                "任何强度下原意、事实、数字、人名、链接都不变"
            ),
            "order": 5,
        },
    )
    rewrite_style: Literal["无", "温柔", "自然", "活泼", "直爽", "软萌", "沉稳", "冷漠", "热情", "自定义"] = Field(
        default="无",
        description="微调风格：微调时优先按所选风格调整措辞与语气（与参考的群聊风格冲突时以本风格为准；原意仍严格不变）；选「自定义」时使用下方自定义描述；注意：选非「无」时，会改为按所选风格微调、跳过按群模型风格的微调（群风格不生效），选「无」才完全按群模型风格微调",
        json_schema_extra={
            "label": "微调风格",
            "hint": (
                "各选项注释："
                + "；".join(f"{name} = {note}" for name, note in REWRITE_STYLE_NOTES.items())
                + "。选非「无」时，风格微调会优先按该风格调整整句话的语气与措辞（原意不变）；"
                "注意：只要选了风格（非「无」），就会改为按所选风格微调、跳过按群模型风格的微调"
                "（群风格不生效），选「无」才完全按群模型风格微调"
            ),
            "order": 5.5,
        },
    )
    rewrite_style_custom: str = Field(
        default="",
        description="自定义微调风格描述（仅当「微调风格」选择「自定义」时生效）：一行文本描述想让微调后的回复用什么语气说话，留空则不注入风格",
        json_schema_extra={
            "label": "自定义微调风格描述",
            "placeholder": "例如：像东北老铁唠嗑、二次元萌妹腔、高冷寡言",
            "hint": "仅当上方风格选择「自定义」时生效；例如填「像东北老铁唠嗑」，微调就会按这个描述塑造回复的语气与措辞；留空 = 不注入任何风格",
            "order": 5.6,
        },
    )
    rewrite_timeout_seconds: float = Field(
        default=60.0,
        description="单次风格微调的最大耗时（秒），超时直接使用原回复",
        json_schema_extra={"label": "微调超时(秒)", "step": 1, "order": 6},
    )
    hook_timeout_seconds: float = Field(
        default=60.0,
        ge=3.0,
        description="微调 Hook 整体超时（秒）：覆盖检索模型、渲染参考文本与 LLM 调用的总耗时上限；放宽微调超时时需同步调大，否则会被整体超时截断",
        json_schema_extra={
            "label": "微调 Hook 超时(秒)",
            "hint": "默认 60。这是 MaiBot 对 Hook 的整体执行上限（含模型检索与参考文本渲染），需 ≥ 微调超时；只调大「微调超时」不调这个会被整体超时截断",
            "step": 1,
            "order": 6.5,
        },
    )
    block_emoji: bool = Field(
        default=False,
        description="禁止微调结果中出现 emoji 表情：微调完成后若发现 emoji 会自动去除（仅作用于微调结果，不影响原回复）",
        json_schema_extra={
            "label": "禁止微调结果中的 emoji",
            "hint": "开启后，微调完成发现结果里有 emoji 表情会直接去掉再发送",
            "order": 7,
        },
    )


class GarbageGuardSection(PluginConfigBase):
    """回复乱码拦截：AI 回复出现乱码、指令残片或内部标识符泄漏时，自动要求回复器重新生成。

    检测在风格微调之后执行（检查最终会发送的文本），命中时回复器会带着
    「重生成约束」重新生成（最多 3 次），重试仍失败则按原逻辑发送最后一次结果。
    """

    __ui_label__ = "乱码拦截"
    __ui_icon__ = "shield-alert"
    __ui_order__ = 25

    enabled: bool = Field(
        default=True,
        description="是否启用回复乱码拦截：检测到乱码/指令残片/内部标识符时要求回复器重新生成",
        json_schema_extra={
            "label": "启用乱码拦截",
            "hint": "默认开启。命中规则时回复器会自动重新生成一条正常回复，通常对用户无感知",
            "order": 0,
        },
    )
    retry_on_detect: bool = Field(
        default=True,
        description="命中乱码时是否触发重新生成；关闭后仅记录日志、不拦截回复",
        json_schema_extra={"label": "命中后重新生成", "order": 1},
    )
    custom_patterns: List[str] = Field(
        default_factory=list,
        description="自定义追加检测正则（一条一行，任一项命中即拦截；正则无效时自动跳过）",
        json_schema_extra={
            "label": "自定义检测正则",
            "placeholder": "一行一条正则，例如 \\bmsg_id\\b",
            "hint": "在内置规则之外追加正则（不区分大小写）。例如想拦截特定词：\\bxxx\\b",
            "order": 2,
        },
    )


class PreRewriteBlockSection(PluginConfigBase):
    """微调前黑名单：AI 生成回复草稿后、风格微调之前，先把草稿中出现的黑名单词删掉，再进行风格微调。

    仅支持中文词条：词条必须包含中文字符才生效，纯英文、数字、符号与 emoji 词条无效（会被忽略）；
    字面匹配采用整词/子串匹配（例如「涩涩」匹配「涩涩」，只含单个字的不会误伤）；
    开启「语义换词匹配」后，会基于没删前的整句话理解意思，删除黑名单词并尽量保持原意，
    补字/换近义时可按「删词补意风格」选择用词倾向（无 / 温柔 / 自然（像正常人聊天）/ 活泼）。
    删除后草稿为空时放弃微调并保留原回复；微调失败或超时时仍返回删词后的文本，
    保证黑名单词不会出现在最终回复里。
    """

    __ui_label__ = "微调前黑名单"
    __ui_icon__ = "filter-x"
    __ui_order__ = 26

    enabled: bool = Field(
        default=False,
        description="是否启用微调前黑名单：草稿中出现黑名单词时，先在微调前删除该词",
        json_schema_extra={"label": "启用微调前黑名单", "order": 0},
    )
    items: List[str] = Field(
        default_factory=list,
        description="微调前黑名单词列表：一条一行，仅支持中文词条（含中文即生效，纯英文/数字/符号/emoji 词条无效）；草稿完整包含任一词时，微调前先删除该词再微调",
        json_schema_extra={
            "label": "黑名单词列表",
            "hint": "一条一行；仅支持中文词条（如「涩涩」），纯英文、数字、符号、emoji 词条无效；草稿中出现「涩涩」时会在风格微调前先删掉（只含单个字的不会误伤）",
            "order": 1,
        },
    )
    semantic_match: bool = Field(
        default=False,
        description="启用语义换词匹配：基于没删前的整句理解意思后删除黑名单词并尽量保持原意，可用近义表达替代（如删「绷不住了」可写「难绷」），只要不出现黑名单词原文即可；每次草稿会额外消耗一次模型调用",
        json_schema_extra={
            "label": "语义换词匹配",
            "hint": "默认关闭，只做字面匹配。开启后：基于没删前的整句话理解意思后再删除，删除后不通顺或丢语气时允许补字/换近义词（如删「属实」补「是」，删「绷不住了」写「难绷」），只要求字面不出现黑名单词；同义换词、变形、谐音表述也会被一并删除；语义换词匹配生效后跳过回复后风格微调（直接返回删词补意后的文本）；仅字面删词命中时仍继续回复后微调；输出异常（仍含黑名单词/长度比例异常）时退回字面删除结果",
            "order": 2,
        },
    )
    rewrite_style: Literal["无", "温柔", "自然", "活泼", "直爽", "软萌", "沉稳", "冷漠", "热情", "自定义"] = Field(
        default="无",
        description="删词补意风格：语义删除黑名单词后，整句话的语气与措辞尽量贴合所选风格（只影响语义换词匹配环节）；选「自定义」时使用下方自定义描述",
        json_schema_extra={
            "label": "删词补意风格",
            "hint": (
                "各选项注释："
                + "；".join(f"{name} = {note}" for name, note in REWRITE_STYLE_NOTES.items())
                + "。选非「无」时，语义删词后的整句话语气措辞会尽量贴合所选风格"
            ),
            "order": 3,
        },
    )
    rewrite_style_custom: str = Field(
        default="",
        description="自定义风格描述（仅当「删词补意风格」选择「自定义」时生效）：一行文本描述想让删词后的句子用什么语气说话，留空则不注入风格",
        json_schema_extra={
            "label": "自定义风格描述",
            "placeholder": "例如：像东北老铁唠嗑、二次元萌妹腔、高冷寡言",
            "hint": "仅当上方风格选择「自定义」时生效；例如填「像东北老铁唠嗑」就会按这个描述塑造删词后整句话的语气与措辞；留空 = 不注入任何风格",
            "order": 4,
        },
    )


class ChatStyleTranslatorConfig(PluginConfigBase):
    """聊天风格转译插件完整配置。"""

    plugin: PluginSection = Field(default_factory=PluginSection)
    owner: OwnerSection = Field(default_factory=OwnerSection)
    storage: StorageSection = Field(default_factory=StorageSection)
    groups: GroupsSection = Field(default_factory=GroupsSection)
    merge: MergeSection = Field(default_factory=MergeSection)
    compact: CompactSection = Field(default_factory=CompactSection)
    user_block: UserBlockSection = Field(default_factory=UserBlockSection)
    collection: CollectionSection = Field(default_factory=CollectionSection)
    analysis: AnalysisSection = Field(default_factory=AnalysisSection)
    blacklist: BlacklistSection = Field(default_factory=BlacklistSection)
    phrase_blacklist: PhraseBlockSection = Field(default_factory=PhraseBlockSection)
    inject: InjectSection = Field(default_factory=InjectSection)
    garbage_guard: GarbageGuardSection = Field(default_factory=GarbageGuardSection)
    pre_rewrite_blacklist: PreRewriteBlockSection = Field(default_factory=PreRewriteBlockSection)


class ChatStyleTranslatorPlugin(MaiBotPlugin):
    """聊天风格转译插件。"""

    config_model = ChatStyleTranslatorConfig

    def __init__(self) -> None:
        """初始化插件实例。"""

        super().__init__()
        self._store: Optional[StyleStore] = None
        self._analyzer: Optional[StyleAnalyzer] = None
        self._loop_task: Optional[asyncio.Task] = None
        self._owner_qq_set: set[str] = set()
        self._session_map: Dict[str, str] = {}
        self._stream_map: Dict[str, str] = {}
        self._analyzing: set[str] = set()
        self._daily_trigger_date: Dict[str, str] = {}
        self._global_models: List[str] = []
        self._config_global_models: List[str] = []
        self._model_dir: Optional[Path] = None
        self._speaker_map: Dict[str, Dict[str, str]] = {}
        self._blocked_users: set[str] = set()
        self._block_slash = True
        self._block_qq_bots = True
        self._bot_ids: set[str] = set()
        self._official_bot_prefixes: tuple[str, ...] = ("285419631",)
        self._rewriting: set[str] = set()
        self._phrase_blacklist: List[str] = []
        self._skip_self_messages = True
        self._self_qq_set: set[str] = set()
        self._self_account_ids: set[str] = set()
        self._pre_rewrite_blacklist: List[str] = []
        self._pre_rewrite_blacklist_enabled = False
        self._pre_rewrite_semantic_match = False
        self._no_symbol_emotion = True
        self._pre_rewrite_style = "无"
        self._pre_rewrite_style_custom = ""
        self._rewrite_style = "无"
        self._rewrite_style_custom = ""
        self._auto_enable_models = True

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def on_load(self) -> None:
        """插件加载：初始化存储、分析器与后台任务。"""

        self._build_store()
        self._analyzer = StyleAnalyzer(self.ctx)
        self._sync_config()
        await self._sync_groups_from_config()
        await self._sync_blacklist_from_config()
        await self._refresh_session_map()
        await self._refresh_global_models()
        await self._maybe_auto_merge()
        await self._maybe_auto_compact()
        self._loop_task = asyncio.get_running_loop().create_task(self._background_loop())
        self._last_manual_trigger_at = 0.0
        self._last_backfill_trigger_at = 0.0
        await self._maybe_manual_backfill_trigger()
        logger.info("聊天风格转译插件已加载，模型目录: %s", self._store.model_dir)

    async def on_unload(self) -> None:
        """插件卸载：清理后台任务。"""

        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        logger.info("聊天风格转译插件已卸载")

    async def on_config_update(self, scope: str, config_data: dict[str, object], version: str) -> None:
        """配置热更新。"""

        del config_data, version
        if scope == "self":
            self._sync_config()
            self._build_store()
            await self._sync_groups_from_config()
            await self._sync_blacklist_from_config()
            await self._refresh_global_models()
            await self._maybe_auto_merge()
            await self._maybe_auto_compact()
            await self._maybe_manual_analyze_trigger()
            await self._maybe_manual_backfill_trigger()
            logger.info("聊天风格转译插件配置已更新，模型目录: %s", self._store.model_dir)

    def _resolve_model_dir(self) -> Optional[Path]:
        """按配置解析模型保存目录；未配置时返回 None（使用默认目录）。"""

        raw = str(self.config.storage.model_dir or "").strip()
        if not raw:
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = self.ctx.paths.data_dir / path
        return path

    def _build_store(self) -> None:
        """按当前配置重建存储对象（模型目录变化时生效）。"""

        model_dir = self._resolve_model_dir()
        if self._store is not None and self._model_dir == model_dir:
            return
        self._model_dir = model_dir
        self._store = StyleStore(self.ctx.paths.data_dir, model_dir=model_dir)

    def _sync_config(self) -> None:
        """把配置同步到内存状态与子模块。"""

        self._no_symbol_emotion = bool(
            getattr(getattr(self.config, "plugin", None), "no_symbol_emotion", True)
        )
        self._auto_enable_models = bool(
            getattr(getattr(self.config, "plugin", None), "auto_enable_models", True)
        )
        self._owner_qq_set = {str(qq).strip() for qq in (self.config.owner.qq or []) if str(qq).strip()}
        self._config_global_models = [
            str(model_id).strip()
            for model_id in (self.config.inject.global_models or [])
            if str(model_id).strip()
        ]
        self._blocked_users = {str(qq).strip() for qq in (self.config.user_block.qq or []) if str(qq).strip()}
        self._block_slash = bool(self.config.user_block.block_slash)
        self._block_qq_bots = bool(self.config.user_block.block_qq_bots)
        self._bot_ids = {str(bot_id).strip() for bot_id in (self.config.user_block.bot_ids or []) if str(bot_id).strip()}
        self._skip_self_messages = bool(getattr(self.config.user_block, "skip_self_messages", True))
        self._self_qq_set = {
            str(qq).strip()
            for qq in (getattr(self.config.user_block, "self_qq", None) or [])
            if str(qq).strip()
        }
        self._phrase_blacklist = [
            str(phrase).strip()
            for phrase in (self.config.phrase_blacklist.items or [])
            if str(phrase).strip()
        ]
        self._pre_rewrite_blacklist_enabled = bool(
            getattr(getattr(self.config, "pre_rewrite_blacklist", None), "enabled", False)
        )
        self._pre_rewrite_blacklist = [
            str(word).strip()
            for word in (getattr(getattr(self.config, "pre_rewrite_blacklist", None), "items", None) or [])
            if str(word).strip()
        ]
        self._pre_rewrite_semantic_match = bool(
            getattr(getattr(self.config, "pre_rewrite_blacklist", None), "semantic_match", False)
        )
        self._pre_rewrite_style = str(
            getattr(getattr(self.config, "pre_rewrite_blacklist", None), "rewrite_style", "") or ""
        ).strip() or "无"
        if self._pre_rewrite_style not in REWRITE_STYLE_GUIDE and self._pre_rewrite_style != "自定义":
            self._pre_rewrite_style = "无"
        self._pre_rewrite_style_custom = str(
            getattr(getattr(self.config, "pre_rewrite_blacklist", None), "rewrite_style_custom", "") or ""
        ).strip()
        self._rewrite_style = str(
            getattr(getattr(self.config, "inject", None), "rewrite_style", "") or ""
        ).strip() or "无"
        if self._rewrite_style not in REWRITE_STYLE_GUIDE and self._rewrite_style != "自定义":
            self._rewrite_style = "无"
        self._rewrite_style_custom = str(
            getattr(getattr(self.config, "inject", None), "rewrite_style_custom", "") or ""
        ).strip()
        if self._analyzer is not None:
            self._analyzer.update_config(
                model=self.config.analysis.model,
                max_statements_per_scene=self.config.analysis.max_statements_per_scene,
            )

    async def _refresh_global_models(self) -> None:
        """刷新全局通用模型列表缓存（配置列表 + 模型文件中的 global_apply 开关）。

        「模型目录自动生效」开启时，额外把模型目录里的全部模型文件自动审核通过并
        纳入全局生效列表，实现「放进目录即对所有群生效」，无需 /风格启用。
        """

        if self._store is None:
            self._global_models = []
            return
        config_globals = []
        for reference in self._config_global_models:
            file_id = self._store.find_model_file(reference)
            if file_id:
                config_globals.append(file_id)
        file_globals = []
        for model in self._store.list_models():
            if not bool(model.get("global_apply")):
                continue
            file_id = self._store.find_model_file(str(model.get("model_id") or ""))
            if file_id:
                file_globals.append(file_id)
        self._global_models = list(dict.fromkeys([*config_globals, *file_globals]))
        if not self._auto_enable_models:
            return
        for file_id in self._store.list_model_ids():
            model = self._store.get_model(file_id)
            if model is None:
                continue
            if str(model.get("status") or "") != "approved":
                model["status"] = "approved"
                try:
                    await self._store.save_model(model)
                except Exception as exc:
                    logger.warning("模型目录自动生效：模型 %s 自动审核通过失败: %s", file_id, exc)
            self._global_models.append(file_id)
        self._global_models = list(dict.fromkeys(self._global_models))

    async def _sync_groups_from_config(self) -> None:
        """把配置里的群列表同步到存储（群绑定 + 该群使用的模型）。

        规则：
        - 配置中的群：自动绑定，并把该群使用的模型（模型名或 ID）写入挂载列表；
        - 曾由配置绑定、但已从配置删除的群：自动解绑；
        - 通过 /风格绑定群 命令绑定的群：不受配置同步影响。
        """

        if self._store is None:
            return
        config_items = {
            str(item.group_id).strip(): item
            for item in self.config.groups.items
            if str(item.group_id or "").strip()
        }
        for group_id, item in config_items.items():
            await self._store.bind_group(
                group_id,
                platform=str(item.platform or "qq").strip() or "qq",
                source="config",
            )
            resolved: List[str] = []
            for reference in item.models or []:
                reference = str(reference).strip()
                if not reference:
                    continue
                model_file = self._store.find_model_file(reference)
                if model_file:
                    resolved.append(model_file)
            await self._store.set_group_field(group_id, "active_models", resolved)
        for group_id, group in list(self._store.list_groups().items()):
            if group.get("source") == "config" and group_id not in config_items:
                await self._store.unbind_group(group_id)
        if config_items:
            logger.info("已同步 %d 个配置群绑定", len(config_items))

    async def _sync_blacklist_from_config(self) -> None:
        """把插件配置里的黑名单条目同步到存储（WebUI 可编辑）。

        配置里的条目（模型名会解析成 ID）写入黑名单；命令添加的条目不受影响。
        """

        if self._store is None:
            return
        resolved: List[Dict[str, Any]] = []
        for item in self.config.blacklist.items or []:
            text = str(item.text or "").strip()
            if not text:
                continue
            reference = str(item.model or "").strip()
            model_id, model_name = "", ""
            if reference:
                model = self._store.find_model(reference)
                if model is not None:
                    model_id = str(model.get("model_id") or "")
                    model_name = str(model.get("name") or reference)
                else:
                    model_name = reference
            resolved.append(
                {
                    "text": text,
                    "meaning": str(item.meaning or "").strip(),
                    "model_id": model_id,
                    "model_name": model_name,
                    "apply_all": bool(item.apply_all),
                }
            )
        await self._store.sync_blacklist_config(resolved)
        if resolved:
            logger.info("已同步 %d 条配置黑名单", len(resolved))

    # ------------------------------------------------------------------
    # 会话映射（session_id <-> group_id）
    # ------------------------------------------------------------------

    async def _refresh_session_map(self) -> None:
        """通过聊天流列表刷新 session_id 与群号的双向映射。"""

        if self._store is None:
            return
        try:
            result = await self.ctx.chat.get_all_streams()
            streams = result.get("streams", []) if isinstance(result, dict) else (result or [])
            if not isinstance(streams, list):
                return
            for stream in streams:
                if not isinstance(stream, dict):
                    continue
                session_id = str(stream.get("session_id") or stream.get("stream_id") or "").strip()
                group_id = str(stream.get("group_id") or "").strip()
                if session_id and group_id:
                    self._remember_session_mapping(session_id, group_id)
                account_id = str(stream.get("account_id") or "").strip()
                if account_id:
                    self._self_account_ids.add(account_id)
        except Exception as exc:
            logger.warning("刷新会话映射失败: %s", exc)

    def _remember_session_mapping(self, session_id: str, group_id: str) -> None:
        """有界记录会话与群的映射（session_id → group_id，含反向映射）。

        先删旧键再赋值以刷新插入顺序，超过 _MAX_SESSION_MAP_ENTRIES 时淘汰最旧的条目，
        并同步清理被淘汰或改绑后残留的反向映射，防止内存无限增长与旧映射套错风格。
        """

        if not session_id or not group_id:
            return
        old_group = self._session_map.get(session_id)
        if old_group and old_group != group_id and self._stream_map.get(old_group) == session_id:
            del self._stream_map[old_group]
        if self._stream_map.get(group_id) == session_id:
            del self._stream_map[group_id]
        if session_id in self._session_map:
            del self._session_map[session_id]
        self._session_map[session_id] = group_id
        while len(self._session_map) > _MAX_SESSION_MAP_ENTRIES:
            oldest_session, oldest_group = next(iter(self._session_map.items()))
            del self._session_map[oldest_session]
            if self._stream_map.get(oldest_group) == oldest_session:
                del self._stream_map[oldest_group]
        if group_id in self._stream_map:
            del self._stream_map[group_id]
        self._stream_map[group_id] = session_id
        while len(self._stream_map) > _MAX_SESSION_MAP_ENTRIES:
            oldest_group, oldest_session = next(iter(self._stream_map.items()))
            del self._stream_map[oldest_group]
            if self._session_map.get(oldest_session) == oldest_group:
                del self._session_map[oldest_session]

    def _remember_message_session(self, message: Dict[str, Any], group_id: str) -> None:
        """从收到的消息记忆 session_id 与群号映射。"""

        session_id = str(message.get("session_id") or message.get("stream_id") or "").strip()
        if session_id and group_id:
            self._remember_session_mapping(session_id, group_id)

    def _group_stream_id(self, group_id: str) -> str:
        """返回绑定群对应的 stream_id（用于发通知），找不到时返回空串。"""

        return self._stream_map.get(group_id, "")

    async def _send_private(self, qq: str, text: str) -> bool:
        """向指定 QQ 私聊发送文本。非好友/发送失败时静默返回 False（只记日志）。"""

        qq = str(qq or "").strip()
        if not qq or not qq.isdigit():
            logger.info("进度私聊：QQ 号无效（%r），跳过发送", qq)
            return False
        try:
            await self.ctx.api.call(
                "adapter.napcat.message.send_msg",
                params={
                    "message_type": "private",
                    "user_id": int(qq),
                    "message": [{"type": "text", "data": {"text": text}}],
                },
            )
            logger.info("进度私聊：已发送到 %s（%d 字）", qq, len(text))
            return True
        except Exception as exc:
            logger.warning("私聊发送失败（可能不是好友）：qq=%s err=%s", qq, exc)
            return False

    async def _notify_analysis(self, group_id: str, text: str) -> None:
        """把分析进度/结果私聊发送给配置的 QQ（未配置或非好友时静默不发送）。"""

        try:
            qq = str(getattr(self.config.analysis, "notify_qq", "") or "").strip()
            if not qq:
                return
            group_name = ""
            if self._store is not None:
                group = self._store.get_group(group_id)
                if group:
                    group_name = str(group.get("group_name") or "")
            prefix = f"【{group_name}】" if group_name else f"【{group_id}】"
            await self._send_private(qq, f"{prefix}{text}")
        except Exception as exc:
            logger.warning("分析进度通知失败: %s", exc)

    async def _notify_merge(self, text: str, prefix: str = "【模型融合】") -> None:
        """把模型融合/精简进度与结果私聊发送给配置的 QQ（未配置或非好友时静默不发送）。"""

        try:
            qq = str(getattr(self.config.analysis, "notify_qq", "") or "").strip()
            if not qq:
                return
            await self._send_private(qq, f"{prefix}{text}")
        except Exception as exc:
            logger.warning("模型融合通知失败: %s", exc)

    # ------------------------------------------------------------------
    # 权限
    # ------------------------------------------------------------------

    def _is_owner(self, kwargs: Dict[str, Any]) -> bool:
        """判断调用者是否为管理员。"""

        if bool(kwargs.get("is_local_operator")):
            return True
        user_id = str(kwargs.get("user_id") or "").strip()
        return bool(user_id) and user_id in self._owner_qq_set

    async def _reply(
        self, kwargs: Dict[str, Any], text: str, intercept: int = 2
    ) -> tuple[bool, str, bool]:
        """发送命令回复文本到当前会话，并返回命令结果元组。

        新版 MaiBot 命令链不再自动发送命令返回文本、只返回结果元组，
        需要插件自己调用发送能力把回复发出去；stream_id 为空（如本地
        操作者）或发送失败时静默跳过，不影响命令返回值。
        """

        stream_id = str(kwargs.get("stream_id") or "").strip()
        if stream_id:
            try:
                await self.ctx.send.text(text, stream_id)
            except Exception as exc:
                logger.warning("命令回复发送失败: %s", exc)
        return True, text, intercept

    async def _deny(self, kwargs: Dict[str, Any], message: str = "权限不足：该命令仅管理员可用。") -> tuple[bool, str, bool]:
        """返回权限不足的响应（并发送到当前会话）。"""

        return await self._reply(kwargs, message)

    # ------------------------------------------------------------------
    # 消息收集
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_group_id(message: Dict[str, Any]) -> str:
        """从消息字典解析群号，非群聊返回空串。"""

        info = message.get("message_info", {})
        if not isinstance(info, dict):
            return ""
        group_info = info.get("group_info") or {}
        if isinstance(group_info, dict):
            return str(group_info.get("group_id") or "").strip()
        return ""

    @staticmethod
    def _extract_sender_id(message: Dict[str, Any]) -> str:
        """从消息字典解析发送者 QQ 号。"""

        info = message.get("message_info", {})
        if not isinstance(info, dict):
            return ""
        user_info = info.get("user_info") or {}
        if isinstance(user_info, dict):
            return str(user_info.get("user_id") or "").strip()
        return ""

    def _is_sender_blocked(self, message: Dict[str, Any]) -> bool:
        """判断发送者是否命中屏蔽配置（用户黑名单 / 机器人 / 官方机器人）。"""

        sender_id = self._extract_sender_id(message)
        if not sender_id:
            return False
        if sender_id in self._blocked_users:
            return True
        if self._block_qq_bots and (
            sender_id in self._bot_ids or sender_id.startswith(self._official_bot_prefixes)
        ):
            return True
        return False

    def _is_self_message(self, message: Dict[str, Any]) -> bool:
        """判断消息是否来自机器人本人（用于跳过收集本人聊天）。

        识别方式（任一命中即视为本人）：
        1. 手动配置的「本人 QQ 号」列表；
        2. 消息自带的机器人账号（additional_config.self_id，NapCat 适配器会写入）与发送者一致；
        3. 已自动识别的机器人账号（additional_config.self_id / 聊天流 account_id）。
        """

        if not self._skip_self_messages:
            return False
        sender_id = self._extract_sender_id(message)
        if not sender_id:
            return False
        if sender_id in self._self_qq_set:
            return True
        if sender_id in self._self_account_ids:
            return True
        info = message.get("message_info", {})
        additional = (info.get("additional_config") or {}) if isinstance(info, dict) else {}
        if isinstance(additional, dict):
            self_id = str(additional.get("self_id") or "").strip()
            if self_id and self_id == sender_id:
                self._self_account_ids.add(self_id)
                return True
        return False

    def _is_text_blocked(self, text: str) -> bool:
        """判断消息文本是否命中屏蔽规则（带 / 的类似命令的句子）。"""

        return bool(self._block_slash and "/" in text)

    def _text_has_blocked_phrase(self, text: str) -> bool:
        """判断消息文本是否完整包含任一字句黑名单。

        整句子串匹配：例如黑名单里有「修仙」，只有文本中完整出现「修仙」才返回 True，
        只含单个「修」字不会误伤。
        """

        if not self._phrase_blacklist:
            return False
        return any(phrase and phrase in text for phrase in self._phrase_blacklist)

    @staticmethod
    def _build_record(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """从消息字典构造收集记录，无有效文本时返回 None。"""

        text = str(message.get("processed_plain_text") or message.get("plain_text") or "").strip()
        if not text or bool(message.get("is_command")):
            return None
        info = message.get("message_info", {})
        if not isinstance(info, dict):
            info = {}
        user_info = info.get("user_info") or {}
        if not isinstance(user_info, dict):
            user_info = {}
        at_user_id, at_user_name = "", ""
        reply_to_id, reply_to_name = "", ""
        for segment in message.get("raw_message") or []:
            if not isinstance(segment, dict):
                continue
            segment_type = str(segment.get("type") or "")
            data = segment.get("data") or {}
            if not isinstance(data, dict):
                data = {}
            if segment_type == "at":
                at_user_id = str(data.get("target_user_id") or "").strip()
                at_user_name = str(
                    data.get("target_user_nickname") or data.get("target_user_cardname") or ""
                ).strip()
            elif segment_type == "reply":
                reply_to_id = str(data.get("target_message_sender_id") or "").strip()
                reply_to_name = str(
                    data.get("target_message_sender_nickname") or data.get("target_message_sender_cardname") or ""
                ).strip()
        return {
            "id": str(message.get("message_id") or ""),
            "ts": time.time(),
            "sender_id": str(user_info.get("user_id") or "").strip(),
            "sender_name": str(
                user_info.get("user_cardname") or user_info.get("user_nickname") or "未知"
            ).strip(),
            "text": text,
            "at_user_id": at_user_id,
            "at_user_name": at_user_name,
            "reply_to_id": reply_to_id,
            "reply_to_name": reply_to_name,
        }

    @HookHandler(
        "chat.receive.after_process",
        name="style_collector",
        description="收集绑定群的群友对话（聊天风格转译插件）",
        mode=HookMode.BLOCKING,
        order=HookOrder.NORMAL,
        timeout_ms=10000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def handle_message(self, message: Any = None, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """异步监听群消息，收集绑定群的对话，并记忆当前说话人（QQ 号）。

        当前 MaiBot 版本不会向插件分发 ON_MESSAGE 事件，因此改用
        ``chat.receive.after_process`` Hook 收集（此时消息已完成轻量预处理，
        processed_plain_text / session_id 已就绪）；返回 None 表示不改写消息。
        """

        del kwargs
        if not isinstance(message, dict) or self._store is None:
            return None
        group_id = self._extract_group_id(message)
        if not group_id:
            return None
        self._remember_message_session(message, group_id)
        if self._is_self_message(message):
            return None
        self._remember_speaker(message)
        if not self._store.is_bound(group_id):
            return None
        if self._is_sender_blocked(message):
            return None
        record = self._build_record(message)
        if record is None:
            return None
        if self._is_text_blocked(record["text"]):
            return None
        if self._text_has_blocked_phrase(record["text"]):
            return None
        count = await self._store.append_message(
            group_id, record, max_stored=int(self.config.collection.max_stored_messages or 0)
        )
        auto_count = self.config.collection.auto_analyze_count
        if auto_count > 0 and count >= auto_count:
            await self._trigger_auto_analyze(group_id, reason="消息条数达到阈值")
        return None

    def _remember_speaker(self, message: Dict[str, Any]) -> None:
        """记忆最近一条群消息的说话人（QQ 号 + 昵称），供回复后微调的说话人匹配使用。"""

        session_id = str(message.get("session_id") or message.get("stream_id") or "").strip()
        if not session_id:
            return
        info = message.get("message_info", {})
        user_info = (info.get("user_info") or {}) if isinstance(info, dict) else {}
        if not isinstance(user_info, dict):
            user_info = {}
        user_id = str(user_info.get("user_id") or "").strip()
        if not user_id:
            return
        self._speaker_map[session_id] = {
            "user_id": user_id,
            "name": str(
                user_info.get("user_cardname") or user_info.get("user_nickname") or "未知"
            ).strip(),
        }
        if len(self._speaker_map) > 500:
            oldest_key = next(iter(self._speaker_map))
            del self._speaker_map[oldest_key]

    # ------------------------------------------------------------------
    # 历史消息回填（NapCat get_group_msg_history）
    # ------------------------------------------------------------------

    @staticmethod
    def _napcat_message_to_record(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """把 NapCat 的群历史消息转换成插件收集记录，无有效文本时返回 None。

        支持 OneBot 11 的 message 段格式：text 拼接为正文，at 记为说话对象，
        reply 记为回复对象，face 记作 [表情]，image/record/video/file 记作 [类型] 占位。
        """

        if not isinstance(raw, dict):
            return None
        sender = raw.get("sender") or {}
        if not isinstance(sender, dict):
            sender = {}
        segments = raw.get("message") or raw.get("messages") or []
        if not isinstance(segments, list):
            segments = []
        text_parts: List[str] = []
        at_user_id, at_user_name = "", ""
        reply_to_id, reply_to_name = "", ""
        has_content = False
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            segment_type = str(segment.get("type") or "")
            if segment_type == "text":
                data = segment.get("data") or {}
                if isinstance(data, dict) and str(data.get("text") or "").strip():
                    has_content = True
            elif segment_type in ("at", "face"):
                has_content = True
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            segment_type = str(segment.get("type") or "")
            data = segment.get("data") or {}
            if not isinstance(data, dict):
                data = {}
            if segment_type == "text":
                text_parts.append(str(data.get("text") or ""))
            elif segment_type == "at":
                target_id = str(data.get("qq") or data.get("user_id") or "").strip()
                if not at_user_id and target_id:
                    at_user_id = target_id
                    at_user_name = str(data.get("name") or "").strip() or f"QQ{target_id}"
                text_parts.append(f"@{at_user_name or at_user_id or '有人'}")
            elif segment_type == "reply":
                if not reply_to_id:
                    reply_to_id = str(
                        data.get("message_id") or data.get("id") or data.get("target_message_sender_id") or ""
                    ).strip()
                reply_name = str(data.get("name") or data.get("target_message_sender_nickname") or "").strip()
                if reply_name:
                    reply_to_name = reply_name
            elif segment_type == "face":
                text_parts.append("[表情]")
            elif segment_type in ("image", "record", "video", "file", "share", "forward", "json"):
                if has_content:
                    text_parts.append(f"[{segment_type}]")
        text = "".join(text_parts).strip()
        if not text or not has_content:
            return None
        sender_id = str(raw.get("user_id") or sender.get("user_id") or "").strip()
        return {
            "id": str(raw.get("message_id") or raw.get("msg_id") or "").strip(),
            "ts": float(raw.get("time") or time.time()),
            "sender_id": sender_id,
            "sender_name": str(sender.get("card") or sender.get("nickname") or "未知").strip(),
            "text": text,
            "at_user_id": at_user_id,
            "at_user_name": at_user_name,
            "reply_to_id": reply_to_id,
            "reply_to_name": reply_to_name,
            "source": "history",
        }

    async def _fetch_group_history(self, group_id: str, count: int, batch_size: int) -> List[Dict[str, Any]]:
        """通过 NapCat ``get_group_msg_history`` 分页拉取指定群的最近历史消息。

        NapCat 的消息 ID 是哈希生成的短 ID（不连续、数值大小与时间无关），
        因此翻页必须以「本页时间最旧的消息」的短 ID 作为 message_seq，
        并配合 ``reverse_order=1`` 向前（更早）取；以「本页是否有新的消息 ID」
        判断是否继续，无推进或调用失败时停止；最多翻页 100 次防止意外死循环。
        过滤规则（用户/机器人屏蔽、带 / 的句子、字句黑名单、本人消息）与实时收集一致。
        """

        records: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        oldest_short_id = ""
        for page in range(100):
            if len(records) >= count:
                break
            params: Dict[str, Any] = {
                "group_id": int(group_id),
                "count": batch_size,
                "reverse_order": 1,
            }
            if oldest_short_id:
                params["message_seq"] = oldest_short_id
            try:
                result = await self.ctx.api.call(
                    "adapter.napcat.message.get_group_msg_history",
                    params=params,
                )
            except Exception as exc:
                logger.warning("拉取群 %s 历史消息失败（第 %d 页，已拉 %d 条）: %s", group_id, page + 1, len(records), exc)
                break
            data = result.get("data") if isinstance(result, dict) else None
            messages = data.get("messages") if isinstance(data, dict) else None
            if not isinstance(messages, list):
                logger.warning("拉取群 %s 历史消息返回结构异常: %s", group_id, str(result)[:300])
                break
            if not messages:
                break
            page_short_ids: List[str] = []
            raw_count = 0
            page_probe = []
            page_oldest_id = ""
            page_oldest_ts = 0.0
            seen_before_page = set(seen_ids)
            for raw in messages:
                if not isinstance(raw, dict):
                    continue
                raw_count += 1
                short_id = str(raw.get("message_seq") or raw.get("message_id") or "").strip()
                if short_id:
                    page_short_ids.append(short_id)
                if len(page_probe) < 3:
                    page_probe.append(
                        {
                            "id": str(raw.get("message_id") or "")[:20],
                            "ts": raw.get("time"),
                            "text": str(
                                ((raw.get("message") or [{}])[0].get("data") or {}).get("text", "")
                                if isinstance(raw.get("message"), list) and raw.get("message")
                                else ""
                            )[:20],
                        }
                    )
                try:
                    raw_ts = float(raw.get("time") or 0)
                except (TypeError, ValueError):
                    raw_ts = 0.0
                if short_id and (not page_oldest_id or raw_ts < page_oldest_ts):
                    page_oldest_id = short_id
                    page_oldest_ts = raw_ts
                mid = str(raw.get("message_id") or "").strip()
                if mid:
                    if mid in seen_ids:
                        continue
                    seen_ids.add(mid)
                record = self._napcat_message_to_record(raw)
                if record is None:
                    continue
                if self._is_text_blocked(record["text"]) or self._text_has_blocked_phrase(record["text"]):
                    continue
                sender_id = record["sender_id"]
                if sender_id:
                    if sender_id in self._blocked_users:
                        continue
                    if self._block_qq_bots and (
                        sender_id in self._bot_ids or sender_id.startswith(self._official_bot_prefixes)
                    ):
                        continue
                    if self._skip_self_messages and (
                        sender_id in self._self_qq_set or sender_id in self._self_account_ids
                    ):
                        continue
                records.append(record)
            if not page_short_ids:
                logger.warning(
                    "拉取群 %s 历史消息：NapCat 返回的消息缺少 message_seq，无法继续翻页（第 %d 页，已拉 %d 条）",
                    group_id,
                    page + 1,
                    len(records),
                )
                break
            new_ids = [short_id for short_id in page_short_ids if short_id not in seen_before_page]
            seen_ids.update(page_short_ids)
            if not new_ids:
                logger.info(
                    "拉取群 %s 历史消息：第 %d 页没有新的消息，翻页结束（已拉 %d 条）",
                    group_id,
                    page + 1,
                    len(records),
                )
                break
            logger.info(
                "拉取群 %s 历史消息第 %d 页：原始 %d 条，累计保留 %d 条（起点 %s，前3条: %s）",
                group_id,
                page + 1,
                raw_count,
                len(records),
                oldest_short_id or "最新",
                json.dumps(page_probe, ensure_ascii=False)[:400],
            )
            oldest_short_id = page_oldest_id or page_short_ids[-1]
        records.sort(key=lambda record: float(record.get("ts") or 0))
        return records

    async def run_backfill(self, group_id: str, count: int = 0) -> str:
        """拉取指定群的历史消息回填到收集池（供手动命令调用），返回结果说明。"""

        return await self._run_backfill(group_id, count=count, notify=False)

    async def _run_backfill(self, group_id: str, count: int = 0, notify: bool = False) -> str:
        """执行一次历史消息回填。

        条数优先取调用参数，其次取配置的「历史回填条数」；拉取结果与已收集消息
        按消息 ID 去重；回填后若收集条数达到自动分析阈值（auto_analyze_count），
        自动触发后台分析。
        """

        if self._store is None:
            return "插件尚未就绪。"
        group = self._store.get_group(group_id)
        if group is None:
            return f"群 {group_id} 尚未绑定。"
        count = int(count or 0)
        if count <= 0:
            count = max(0, int(getattr(self.config.collection, "backfill_count", 0) or 0))
        if count <= 0:
            return "未指定拉取条数，且配置里的「历史回填条数」为 0。"
        batch_size = max(1, min(100, int(getattr(self.config.collection, "backfill_batch_size", 50) or 50)))
        before = self._store.message_count(group_id)
        if notify:
            await self._notify_analysis(
                group_id,
                f"开始历史回填：拉取该群最近 {count} 条消息（单批 {batch_size} 条，向前翻页）。",
            )
        try:
            records = await self._fetch_group_history(group_id, count, batch_size)
        except Exception as exc:
            logger.exception("群 %s 历史回填失败", group_id)
            result = f"群 {group_id} 历史回填失败: {exc}"
            if notify:
                await self._notify_analysis(group_id, result)
            return result
        if not records:
            result = (
                f"群 {group_id} 未拉取到历史消息：请确认机器人已加入该群、NapCat 在线，"
                "且该群允许获取历史记录（部分群/临时会话拉不到历史）。"
            )
            if notify:
                await self._notify_analysis(group_id, result)
            return result
        new_count = await self._store.append_messages(
            group_id, records, max_stored=int(self.config.collection.max_stored_messages or 0)
        )
        added = new_count - before
        await self._store.set_group_field(group_id, "last_backfilled_at", time.time())
        result = f"已从 NapCat 拉取群 {group_id} 历史消息 {len(records)} 条（新增 {added} 条，现有 {new_count} 条）。"
        auto_count = self.config.collection.auto_analyze_count
        if auto_count > 0 and new_count >= auto_count:
            result += " 收集条数已达阈值，已触发自动分析。"
            await self._trigger_auto_analyze(group_id, reason="历史回填后达到阈值")
        else:
            result += " 用 /风格生成 执行分析即可。"
        if notify:
            await self._notify_analysis(group_id, result)
        return result

    async def _maybe_manual_backfill_trigger(self) -> None:
        """WebUI「立即执行历史回填」开关：打开并保存配置后，对全部绑定群触发一次历史回填。

        带 30 秒防抖，避免同一次开启被重复触发。
        """

        try:
            if not bool(getattr(self.config.collection, "trigger_backfill_now", False)):
                return
            now = time.time()
            if now - getattr(self, "_last_backfill_trigger_at", 0.0) < 30:
                return
            self._last_backfill_trigger_at = now
            if self._store is None:
                return
            group_ids = [str(gid) for gid in self._store.list_groups().keys()]
            if not group_ids:
                logger.info("「立即执行历史回填」已触发，但没有已绑定的群")
                return
            for group_id in group_ids:
                await self._run_backfill(group_id, notify=True)
            logger.info("「立即执行历史回填」已触发 %d 个群", len(group_ids))
        except Exception as exc:
            logger.error("「立即执行历史回填」触发失败: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # 回复后轻度微调（不改变 AI 自身行为风格，只改几个词、原意不变）
    # ------------------------------------------------------------------

    def _resolve_active_models(self, group_id: str) -> List[str]:
        """解析当前群生效的风格模型 ID 列表：开启群聊通用时用通用模型，否则用该群挂载的模型。"""

        if self._global_models:
            return list(self._global_models)
        group = self._store.get_group(group_id)
        if not group:
            return []
        mounted = group.get("active_models") or []
        if not isinstance(mounted, list):
            mounted = []
        return [str(model_id) for model_id in mounted if str(model_id).strip()]

    @staticmethod
    def _strip_rewrite_wrappers(text: str) -> str:
        """去掉 LLM 常见的包裹装饰（代码围栏、前缀说明、成对引号）。"""

        text = str(text or "").strip()
        text = re.sub(r"^```(?:text|plain)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE)
        for prefix in ("微调后：", "改写后：", "润色后：", "微调结果：", "结果：", "回复："):
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
                break
        if len(text) >= 2 and text[0] in "\"'「『“" and text[-1] in "\"'」』”":
            inner = text[1:-1].strip()
            if inner:
                text = inner
        return text

    @staticmethod
    def _rewrite_passes_safety(original: str, rewritten: str) -> bool:
        """微调结果安全检查：非空、长度比例在合理范围内（防止 LLM 增删大量内容）。"""

        rewritten = rewritten.strip()
        if not rewritten:
            return False
        ratio = len(rewritten) / len(original)
        return 0.5 <= ratio <= 2.0

    @staticmethod
    def _strip_emoji(text: str) -> str:
        """去除文本中的 emoji 表情（含肤色修饰符、零宽连接序列与变体选择符）。"""

        if not text:
            return text
        emoji_pattern = re.compile(
            "["
            "\U0001F000-\U0001FAFF"  # 补充符号与象形文字
            "\U0001F1E6-\U0001F1FF"  # 区域指示符（国旗）
            "\U00002600-\U000027BF"  # 杂项符号 / 装饰符号 / 箭头
            "\U000023E9-\U000023FA"  # 电视/媒体控制符号
            "\U00002B00-\U00002BFF"  # 杂项符号和箭头
            "\U0000FE0F"  # 变体选择符-16
            "\U0000200D"  # 零宽连接符
            "\U0001F3FB-\U0001F3FF"  # 肤色修饰符
            "]+",
            flags=re.UNICODE,
        )
        cleaned = emoji_pattern.sub("", text)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip()

    @staticmethod
    def _contains_cjk(word: str) -> bool:
        """判断词条是否包含中文字符（含 CJK 扩展 A 区）。

        微调前黑名单仅支持中文词条：词条必须包含中文才生效，
        纯英文、数字、符号与 emoji 词条一律无效（不参与字面匹配与语义匹配）。
        """

        return any("\u3400" <= ch <= "\u9fff" for ch in word)

    @staticmethod
    def _clean_after_strip(text: str) -> str:
        """删除词条后的文本清理：压缩多余空白、合并重复标点、清理残留的首尾标点。"""

        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"([，。！？、；：,.!?;:])[\s，。！？、；：,.!?;:]*\1", r"\1", text)
        return text.strip(" \t\n，。！？、；：,.!?;:")

    def _strip_pre_rewrite_blacklist(self, text: str) -> tuple[str, bool]:
        """微调前黑名单（字面匹配）：删除回复草稿中包含的黑名单词，返回清理后的文本与是否命中。

        仅支持中文词条（含中文字符即生效，纯英文/数字/符号/emoji 词条被忽略）；
        整词/子串匹配：例如黑名单里填了「涩涩」，草稿中出现「涩涩」就会被删掉，
        只含单个「涩」字不会误伤。黑名单未启用、无有效中文词条或未命中时不做改动。
        """

        text = str(text or "")
        if not text or not self._pre_rewrite_blacklist_enabled:
            return text, False
        active_words = [word for word in self._pre_rewrite_blacklist if self._contains_cjk(word)]
        if not active_words:
            return text, False
        cleaned = text
        for word in active_words:
            cleaned = cleaned.replace(word, "")
        if cleaned == text:
            return text, False
        cleaned = self._clean_after_strip(cleaned)
        return cleaned, True

    def _style_guide(self, style: str, custom: str) -> str:
        """返回风格对应的 LLM 提示词要求；无风格或自定义描述为空时返回空串（不注入）。

        预设风格查 REWRITE_STYLE_GUIDE 表；「自定义」使用自定义描述文本；
        选「自定义」但描述留空时视为无风格。
        """

        style = style or "无"
        if style == "自定义":
            custom = (custom or "").strip()
            if not custom:
                return ""
            return f"输出整句话要按照以下描述塑造语气与措辞：{custom}"
        return REWRITE_STYLE_GUIDE.get(style, "")

    def _pre_rewrite_style_guide(self) -> str:
        """返回当前「删词补意风格」对应的 LLM 提示词要求；无风格或自定义为空时返回空串（不注入）。"""

        return self._style_guide(self._pre_rewrite_style, self._pre_rewrite_style_custom)

    def _rewrite_style_guide(self) -> str:
        """返回当前「微调风格」对应的 LLM 提示词要求；无风格或自定义为空时返回空串（不注入）。"""

        return self._style_guide(self._rewrite_style, self._rewrite_style_custom)

    async def _semantic_remove_keep_meaning(self, original: str, literal_cleaned: str, words: List[str]) -> str:
        """语义换词匹配（保持原意）：基于「没删前的整句话」理解意思，删除黑名单词条及其同义换词表述，
        并尽量保持原句子的意思——删除造成语气/意思缺失时可补充字词（例如删「属实」后可补「是」，
        让「这波属实躺赢进世界赛了啊」变成「这波是躺赢进世界赛了啊」，意思不变）。

        输入为「没删前的原句」与「字面删除后的草稿」，输出删除后尽量保持原意的完整句子；
        输出仍含词条字面、长度比例异常（低于原文 50% 或超过 130%）、LLM 失败或超时时退回字面删除结果。
        开启后每条草稿会额外消耗一次模型调用（复用「微调模型任务名」）。
        """

        active_words = [word for word in words if self._contains_cjk(word)]
        if not self._pre_rewrite_blacklist_enabled or not original or not active_words:
            return literal_cleaned
        system = (
            "你是一个文本删除助手。用户会给出「目标词条列表」、「原草稿」和「已按词条删除后的草稿」。\n"
            "任务：输出删除所有目标词条之后的草稿，并尽量保持原句子的意思。\n"
            "要求：\n"
            "1. 先理解整句话的意思再删除，不要机械删字：如果删除后句子不通顺或丢失了语气/强调意思，"
            "可以补充必要的字词使其通顺且意思不变（例如删除「属实」后可在原位置补「是」，"
            "让「这波属实躺赢进世界赛了啊」变成「这波是躺赢进世界赛了啊」，不改变原意）；\n"
            "2. 目标词条原文一律不得出现在输出中；可以自由使用近义表达替代被删除词的意思"
            "（例如删除「绷不住了」后可写「难绷」，删除「属实」后可写「确实」），只要字面不出现目标词条即可；\n"
            "3. 除删除与必要的补字/换近义外，其他内容不要改动；\n"
            "4. 选择与整体语气风格匹配的近义用词：网络梗词、流行语仅在该风格合适时使用，风格明确禁止时不得使用；\n"
            "5. 如果草稿中没有任何目标词条或其同义表述，就原样输出整个草稿；\n"
            "6. 只输出删除后的草稿本身，不要任何解释、前缀、引号或编号。"
        )
        style_guide = self._pre_rewrite_style_guide()
        if style_guide:
            system += f"\n7. 风格要求：{style_guide}。"
        if self._no_symbol_emotion:
            system += f"\n8. {NO_SYMBOL_EMOTION_RULE}。"
        prompt = (
            f"【目标词条列表】\n{chr(10).join('- ' + word for word in active_words)}\n\n"
            f"【原草稿】\n{original}\n\n"
            f"【已按词条删除后的草稿（供参考）】\n{literal_cleaned}\n\n"
            "请按系统要求输出删除并尽量保持原意后的草稿。"
        )
        timeout = max(3.0, float(getattr(self.config.inject, "rewrite_timeout_seconds", None) or 60.0))
        try:
            result = await asyncio.wait_for(
                self.ctx.llm.generate(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    model=str(getattr(self.config.inject, "rewrite_model", None) or "").strip(),
                    temperature=0.0,
                    rpc_timeout_ms=120000,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("微调前黑名单语义删除超时（%.0fs），退回字面删除结果", timeout)
            return literal_cleaned
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("微调前黑名单语义删除失败，退回字面删除结果: %s", exc)
            return literal_cleaned
        if not isinstance(result, dict) or not result.get("success"):
            logger.warning("微调前黑名单语义删除未返回有效结果，退回字面删除结果")
            return literal_cleaned
        raw = self._strip_rewrite_wrappers(str(result.get("response") or ""))
        if not raw or raw == "无":
            return literal_cleaned
        if any(word in raw for word in active_words):
            logger.info("微调前黑名单语义删除：输出仍含黑名单词，退回字面删除结果")
            return literal_cleaned
        ratio = len(raw) / len(original) if original else 1.0
        if ratio < 0.5 or ratio > 1.3:
            logger.info(
                "微调前黑名单语义删除：输出长度比例异常（%.2f，%d 字 → %d 字），退回字面删除结果",
                ratio, len(original), len(raw),
            )
            return literal_cleaned
        return raw

    async def _rewrite_with_llm(
        self,
        reference: str,
        response: str,
        original_response: str = "",
        forbidden_words: Optional[List[str]] = None,
    ) -> Optional[str]:
        """调用 LLM 对回复草稿做风格微调（强度由 rewrite_strength 控制），返回改写后的文本；任何失败返回 None。

        命中微调前黑名单时会传入「删除前的原句」与黑名单词：微调 LLM 可参考原句保持句子意思，
        但不得恢复或引入黑名单词（原句仅作意思参考）。
        """

        strength = max(0, min(100, int(getattr(self.config.inject, "rewrite_strength", 30) or 30)))
        if strength <= 0:
            return None
        if strength < 25:
            degree = "最轻度：只允许调整语气词、助词、标点或个别口语用词，改动幅度必须极小（一般不超过一两个词）"
        elif strength < 45:
            degree = "轻度：只允许调整少量用词、语气词、标点或口语化表达，改动幅度必须极小（一般不超过两三个词或标点）"
        elif strength < 65:
            degree = "中等：可以适当调整用词、语气词、句式和口语化表达，让句子更贴近群聊风格，但仍只做措辞层面的调整"
        elif strength < 85:
            degree = "较强：可以更明显地调整用词、语气、句式和口语化表达，让句子明显贴近群聊风格，但仍不得改变原意"
        else:
            degree = "强烈：可以大胆调整用词、语气、句式和口语化表达，让句子充分贴合群聊风格，但必须保持原意严格不变"
        system = (
            "你是一名群聊风格微调助手。参考给定的群聊风格资料，对用户给出的 AI 回复草稿做风格微调。\n"
            f"本次微调强度：{degree}。\n"
            "铁律（必须严格遵守）：\n"
            "1. 原意绝不能变：句子的意思、信息、事实、数字、专有名词、人名、链接一律不得改动或增减；\n"
            "2. 只做措辞层面的调整：可以换用词、调语气、改句式，但禁止添加、删除、合并、拆分任何实质内容，禁止解释或引用原文；\n"
            "3. 如果草稿已经符合群聊风格或无法安全微调，则一个字符都不改、原样输出；\n"
            "4. 只输出微调后的文本本身，不要任何前缀、后缀、解释、引号或代码块标记。"
        )
        if original_response and original_response != response and forbidden_words:
            system += (
                "\n5. 该草稿已按黑名单删除以下词句：" + "、".join(forbidden_words)
                + "；草稿已尽量保持原意，微调时不得恢复或引入这些黑名单词原文（近义表达可用，"
                "例如可写「难绷」但不能写「绷不住了」），也不要参考原句把这些词写回来。"
            )
        style_guide = self._rewrite_style_guide()
        if style_guide:
            system += (
                f"\n风格要求（本次微调优先按照此风格调整措辞与语气；与群聊风格资料冲突时以本风格为准，"
                f"仍必须遵守上述铁律，不得改变原意）：{style_guide}"
            )
        if self._no_symbol_emotion:
            system += f"\n{NO_SYMBOL_EMOTION_RULE}。"
        prompt = f"【群聊风格资料】\n{reference}\n\n【AI 回复草稿】\n{response}"
        if original_response and original_response != response:
            prompt += (
                "\n\n【删除前的原句（仅供理解意思，不得恢复其中的黑名单词）】\n" + original_response
            )
        prompt += "\n\n请按系统要求输出微调后的回复。"
        timeout = max(3.0, float(getattr(self.config.inject, "rewrite_timeout_seconds", None) or 60.0))
        try:
            result = await asyncio.wait_for(
                self.ctx.llm.generate(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    model=str(getattr(self.config.inject, "rewrite_model", None) or "").strip(),
                    temperature=0.2,
                    rpc_timeout_ms=150000,  # 微调耗时由 rewrite_timeout_seconds 控制，RPC 超时放宽避免提前中断
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("风格微调 LLM 调用超时（%.0fs），使用原回复", timeout)
            return None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("风格微调 LLM 调用失败: %s", exc)
            return None
        if not isinstance(result, dict) or not result.get("success"):
            logger.warning("风格微调 LLM 未返回有效结果，使用原回复")
            return None
        return self._strip_rewrite_wrappers(str(result.get("response") or ""))

    @HookHandler(
        "maisaka.replyer.after_response",
        name="style_rewrite",
        description="AI 生成回复后按群聊风格轻度微调几个词（原意不变，不改变 AI 自身行为风格）",
        mode=HookMode.BLOCKING,
        order=HookOrder.NORMAL,
        timeout_ms=300000,  # 兜底上限：实际整体超时由配置 inject.hook_timeout_seconds 控制
        error_policy=ErrorPolicy.SKIP,
    )
    async def rewrite_style_response(self, **kwargs: Any) -> Dict[str, Any]:
        """在模型生成回复草稿后，检索该群风格模型并让 LLM 做轻度微调（只改几个词）。

        不再把风格模型注入 AI 的提示词，因此 AI 聊天时的自身行为风格保持不变；
        仅在草稿生成后按群风格微调措辞，原意严格不变；任何失败都退回原回复。
        整体耗时受「微调 Hook 超时」配置限制（含模型检索与参考文本渲染），
        超时或失败时跳过微调、返回原回复。
        """

        timeout = max(3.0, float(getattr(self.config.inject, "hook_timeout_seconds", None) or 60.0))
        try:
            return await asyncio.wait_for(
                self._rewrite_style_response_inner(**kwargs), timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.warning("风格微调 Hook 整体超时（%.0fs），跳过微调使用原回复", timeout)
            return {"action": "continue"}

    async def _rewrite_style_response_inner(self, **kwargs: Any) -> Dict[str, Any]:
        """微调 Hook 实际执行体（微调前黑名单去词 + 参考文本检索 + LLM 微调 + 安全检查）。"""

        if not self.config.inject.enabled or self._store is None:
            return {"action": "continue"}
        original_response = str(kwargs.get("response") or "").strip()
        session_id = str(kwargs.get("session_id") or "").strip()
        if not original_response or len(original_response) < 4 or not session_id:
            return {"action": "continue"}
        if session_id in self._rewriting:
            return {"action": "continue"}
        # 立即标记会话正在微调（检查与加入之间不得出现 await），
        # 避免同会话并发任务重复微调；后续所有路径统一由 finally 释放
        self._rewriting.add(session_id)
        try:
            # 微调前黑名单：先删除草稿中出现的黑名单词，再进行风格微调
            # 1) 字面匹配（仅中文词条生效）；2) 开启「语义换词匹配」时，基于没删前的整句话做保持原意的智能删除
            response, _ = self._strip_pre_rewrite_blacklist(original_response)
            semantic_applied = False
            if (
                self._pre_rewrite_blacklist_enabled
                and response
                and self._pre_rewrite_semantic_match
                and self._pre_rewrite_blacklist
            ):
                semantic_cleaned = await self._semantic_remove_keep_meaning(
                    original_response, response, self._pre_rewrite_blacklist
                )
                if semantic_cleaned and semantic_cleaned != response:
                    response = semantic_cleaned
                    semantic_applied = True
            blacklist_hit = response != original_response
            if blacklist_hit:
                logger.info(
                    "微调前黑名单：已从回复草稿中删除黑名单词（会话 %s，%d 字 → %d 字）",
                    session_id, len(original_response), len(response),
                )
                if not response:
                    logger.warning("微调前黑名单：删除黑名单词后草稿为空，放弃微调并保留原回复")
                    return {"action": "continue"}
                if semantic_applied:
                    # 语义换词匹配已生效：删词补意后的句子已自带完整意思与语气，
                    # 不再叠加回复后风格微调，直接返回删词后的文本（避免二次改写失真）
                    logger.info(
                        "语义换词匹配已生效，跳过回复后风格微调（会话 %s）",
                        session_id,
                    )
                    return self._blacklist_only_result(kwargs, original_response, response, blacklist_hit)

            group_id = self._session_map.get(session_id)
            if not group_id:
                return self._blacklist_only_result(kwargs, original_response, response, blacklist_hit)
            active_ids = self._resolve_active_models(group_id)
            if not active_ids:
                return self._blacklist_only_result(kwargs, original_response, response, blacklist_hit)
            speaker = self._speaker_map.get(session_id) if self.config.inject.enable_style_matching else None
            references = []
            for model_id in active_ids:
                model = self._store.get_model(model_id)
                if not model:
                    continue
                reference = build_rewrite_reference(
                    model,
                    max_chars=self.config.inject.max_chars,
                    speaker=speaker,
                    enable_matching=self.config.inject.enable_style_matching,
                )
                if reference:
                    references.append(reference)
            if not references:
                return self._blacklist_only_result(kwargs, original_response, response, blacklist_hit)
            if blacklist_hit:
                # 微调 LLM 的「不得恢复」提示只列出有效中文词条，避免无效词条（纯英文/emoji）误导
                effective_words = [
                    word for word in self._pre_rewrite_blacklist if self._contains_cjk(word)
                ]
                rewritten = await self._rewrite_with_llm(
                    "\n\n".join(references),
                    response,
                    original_response=original_response,
                    forbidden_words=effective_words,
                )
            else:
                rewritten = await self._rewrite_with_llm("\n\n".join(references), response)
            if rewritten:
                # 微调结果兜底：若仍出现黑名单词（LLM 恢复/引入），再删一次
                rewritten, rescan_hit = self._strip_pre_rewrite_blacklist(rewritten)
                if rescan_hit:
                    logger.info("微调结果仍含黑名单词，已兜底删除")
                    if not rewritten:
                        rewritten = None
            if not rewritten or not self._rewrite_passes_safety(response, rewritten):
                return self._blacklist_only_result(kwargs, original_response, response, blacklist_hit)
            if self.config.inject.block_emoji:
                rewritten = self._strip_emoji(rewritten)
                if not rewritten or not self._rewrite_passes_safety(response, rewritten):
                    return self._blacklist_only_result(kwargs, original_response, response, blacklist_hit)
            if rewritten == response:
                return self._blacklist_only_result(kwargs, original_response, response, blacklist_hit)
            modified_kwargs = dict(kwargs)
            modified_kwargs["response"] = rewritten
            return {"action": "continue", "modified_kwargs": modified_kwargs}
        finally:
            self._rewriting.discard(session_id)

    def _blacklist_only_result(
        self,
        kwargs: Dict[str, Any],
        original: str,
        cleaned: str,
        blacklist_hit: bool,
    ) -> Dict[str, Any]:
        """微调不可用/失败时的兜底返回：命中微调前黑名单时返回删词后的文本，否则不改动。"""

        if blacklist_hit and cleaned and cleaned != original:
            modified_kwargs = dict(kwargs)
            modified_kwargs["response"] = cleaned
            return {"action": "continue", "modified_kwargs": modified_kwargs}
        return {"action": "continue"}

    @HookHandler(
        "maisaka.replyer.after_response",
        name="garbage_guard",
        description="检测回复中的乱码/指令残片/内部标识符，命中时要求回复器重新生成",
        mode=HookMode.BLOCKING,
        order=HookOrder.LATE,
        timeout_ms=3000,
        error_policy=ErrorPolicy.SKIP,
    )
    async def guard_garbage_response(self, **kwargs: Any) -> Dict[str, Any]:
        """在风格微调之后检查最终回复文本，命中乱码规则时请求回复器重新生成。

        返回 retry=True 时，回复器会带着「重生成约束」重试（最多 3 次）；
        重试仍失败则按回复器原逻辑发送最后一次结果。关闭重生成时仅记录日志。
        """

        if not self.config.garbage_guard.enabled:
            return {"action": "continue"}
        response = str(kwargs.get("response") or "").strip()
        if not response or len(response) < 4:
            return {"action": "continue"}
        try:
            match = detect_garbage(
                response,
                extra_patterns=self.config.garbage_guard.custom_patterns,
            )
        except Exception as exc:
            logger.warning("乱码检测执行异常，跳过拦截: %s", exc)
            return {"action": "continue"}
        if match is None:
            return {"action": "continue"}
        logger.warning(
            "乱码拦截命中: 规则=%s 描述=%s 会话=%s 命中片段=%r 回复=%r",
            match.rule.name,
            match.rule.description,
            str(kwargs.get("session_id") or ""),
            match.matched_text,
            response[:300],
        )
        if not self.config.garbage_guard.retry_on_detect:
            return {"action": "continue"}
        modified_kwargs = dict(kwargs)
        modified_kwargs["retry"] = True
        modified_kwargs["retry_reason"] = (
            f"回复疑似乱码或指令残片（{match.rule.name}：{match.rule.description}）"
        )
        modified_kwargs["matched_regex"] = match.rule.name
        modified_kwargs["matched_regex_pattern"] = match.rule.pattern
        modified_kwargs["matched_regex_description"] = match.rule.description
        return {"action": "continue", "modified_kwargs": modified_kwargs}

    # ------------------------------------------------------------------
    # 后台分析任务
    # ------------------------------------------------------------------

    async def _maybe_manual_analyze_trigger(self) -> None:
        """WebUI「立即执行分析」开关：打开并保存配置后，对全部绑定群触发一次后台分析。

        带 30 秒防抖，避免同一次开启被重复触发；分析进度可通过 /风格状态 查看。
        """

        try:
            if not bool(self.config.collection.trigger_analyze_now):
                return
            now = time.time()
            if now - getattr(self, "_last_manual_trigger_at", 0.0) < 30:
                return
            self._last_manual_trigger_at = now
            if self._store is None:
                return
            group_ids = [str(gid) for gid in self._store.list_groups().keys()]
            if not group_ids:
                logger.info("「立即执行分析」已触发，但没有已绑定的群")
                return
            for group_id in group_ids:
                await self._trigger_auto_analyze(group_id, reason="WebUI 按钮触发")
            logger.info("「立即执行分析」已触发 %d 个群的后台分析", len(group_ids))
        except Exception as exc:
            logger.error("「立即执行分析」触发失败: %s", exc, exc_info=True)

    async def _background_loop(self) -> None:
        """后台任务：定期刷新会话映射、检查时间点触发与收集阈值。"""

        while True:
            await asyncio.sleep(60)
            try:
                await self._refresh_session_map()
                await self._refresh_global_models()
                await self._check_scheduled_analyze()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("后台任务异常: %s", exc)

    async def _check_scheduled_analyze(self) -> None:
        """检查是否到达每日分析时间点或收集阈值。"""

        if self._store is None:
            return
        daily_time = str(self.config.collection.daily_time or "").strip()
        now = datetime.now()
        auto_count = self.config.collection.auto_analyze_count
        max_stored = int(self.config.collection.max_stored_messages or 0)
        for group_id in list(self._store.list_groups().keys()):
            if max_stored > 0 and self._store.message_count(group_id) > max_stored:
                await self._store.truncate_messages(group_id, max_stored)
            if group_id in self._analyzing:
                continue
            if auto_count > 0 and self._store.message_count(group_id) >= auto_count:
                await self._trigger_auto_analyze(group_id, reason="消息条数达到阈值")
                continue
            if daily_time:
                today = now.strftime("%Y-%m-%d")
                if self._daily_trigger_date.get(group_id) == today:
                    continue
                if now.strftime("%H:%M") >= daily_time:
                    self._daily_trigger_date[group_id] = today
                    await self._trigger_auto_analyze(group_id, reason="到达每日分析时间")

    async def _trigger_auto_analyze(self, group_id: str, reason: str) -> None:
        """触发后台分析（带并发保护）。"""

        if group_id in self._analyzing:
            return
        if self._store is None:
            return
        asyncio.get_running_loop().create_task(self._run_analysis(group_id, reason, notify=True))

    async def run_analysis(self, group_id: str) -> str:
        """执行一次完整的转译分析（供手动命令调用），返回结果说明。"""

        return await self._run_analysis(group_id, reason="手动触发", notify=False)

    async def _report_analysis_failure(self, group_id: str, notify: bool, text: str) -> str:
        """分析失败统一处理：记录日志，notify 开启时私聊通知失败原因。"""

        logger.warning("分析失败 群=%s: %s", group_id, text)
        if notify:
            await self._notify_analysis(group_id, f"分析失败：{text}")
        return text

    async def _run_analysis(self, group_id: str, reason: str, notify: bool = False) -> str:
        """执行一次完整的转译分析。"""

        if self._store is None or self._analyzer is None:
            return "插件尚未就绪。"
        if group_id in self._analyzing:
            return f"群 {group_id} 正在分析中，请稍后再试。"
        group = self._store.get_group(group_id)
        if group is None:
            return f"群 {group_id} 尚未绑定。"
        self._analyzing.add(group_id)
        logger.info("开始转译分析 群=%s 原因=%s", group_id, reason)
        try:
            messages = self._store.messages_for(group_id)
            max_messages = max(1, int(self.config.collection.max_messages))
            window_size = max(5, int(self.config.collection.window_size))
            if len(messages) > max_messages:
                messages = messages[-max_messages:]
            if self._phrase_blacklist:
                messages = [
                    message
                    for message in messages
                    if not self._text_has_blocked_phrase(str(message.get("text") or ""))
                ]
            if self._skip_self_messages:
                self_ids = self._self_qq_set | self._self_account_ids
                if self_ids:
                    messages = [
                        message
                        for message in messages
                        if str(message.get("sender_id") or "") not in self_ids
                    ]
            if not messages:
                return await self._report_analysis_failure(
                    group_id,
                    notify,
                    f"群 {group_id} 还没有收集到可分析的对话。",
                )
            await self._store.set_group_field(group_id, "analyzing", True)
            if notify:
                await self._notify_analysis(
                    group_id,
                    f"开始分析（{reason}）：共 {len(messages)} 条消息，将分批用 LLM 分析。",
                )

            partials = []
            total_messages = len(messages)
            windows = [messages[start : start + window_size] for start in range(0, total_messages, window_size)]
            concurrency = max(1, min(4, int(getattr(self.config.analysis, "analyze_concurrency", 2) or 1)))
            progress_interval = max(0.0, float(getattr(self.config.analysis, "progress_interval_seconds", 120.0) or 0.0))

            # 有限并发分析各批（并发不改变分析内容，仅加快速度；结果按原始顺序存放）
            # 单批失败只记录日志并跳过该批，不取消其他批次（部分结果仍可正常融合）
            semaphore = asyncio.Semaphore(concurrency)
            batch_results: List[Optional[Dict[str, Any]]] = [None] * len(windows)
            processed_count = 0
            last_progress_sent = 0.0

            async def _analyze_batch(index: int, window: List[Dict[str, Any]]) -> None:
                nonlocal processed_count, last_progress_sent
                try:
                    async with semaphore:
                        batch_results[index] = await self._analyzer.analyze_window(window)
                except Exception as exc:
                    logger.warning("分析批次 %d/%d 失败（跳过该批，不影响其他批）: %s", index + 1, len(windows), exc)
                finally:
                    processed_count += len(window)
                    # 总进度通知：首批完成后必发一次，之后按时间间隔节流（间隔为 0 时每批都发）
                    now = time.time()
                    if notify and (progress_interval <= 0 or last_progress_sent == 0
                                   or now - last_progress_sent >= progress_interval):
                        last_progress_sent = now
                        pct = int(processed_count * 100 / total_messages) if total_messages else 0
                        await self._notify_analysis(
                            group_id,
                            f"分析进度：已处理 {processed_count}/{total_messages} 条（{pct}%）",
                        )

            await asyncio.gather(
                *(_analyze_batch(i, w) for i, w in enumerate(windows))
            )
            partials = [p for p in batch_results if p is not None]
            if not partials:
                return await self._report_analysis_failure(
                    group_id,
                    notify,
                    f"群 {group_id} 分析失败：LLM 未返回有效结果（请检查分析模型配置与账号余额）。",
                )
            model = await self._analyzer.merge_partials(
                partials,
                name=f"{group.get('group_name') or group_id} 风格模型",
                options=self._merge_options_from_config(),
            )
            if model is None:
                return await self._report_analysis_failure(
                    group_id,
                    notify,
                    f"群 {group_id} 分析失败：模型融合失败。",
                )

            blacklist = self._store.list_blacklist()
            dropped_count = 0
            threshold = float(self.config.blacklist.similarity_threshold)
            model_id = str(model.get("model_id") or "")
            for scene in model.get("scenes") or []:
                kept, dropped = await self._analyzer.filter_statements(
                    scene.get("statements") or [],
                    blacklist,
                    threshold,
                    model_id=model_id,
                )
                scene["statements"] = kept
                dropped_count += len(dropped)

            model["model_id"] = self._store.new_model_id()
            model["group_id"] = group_id
            model["status"] = "pending"
            model["created_at"] = time.time()
            model["dropped_by_blacklist"] = dropped_count
            await self._store.save_model(model)
            await self._store.set_group_field(group_id, "last_analyzed_at", time.time())
            await self._store.clear_messages(group_id)
            result = (
                f"分析完成，已生成模型「{model.get('name')}」（{len(model.get('scenes') or [])} 个场景，"
                f"黑名单过滤 {dropped_count} 条语句）。请用 /风格详情 审核，确认无误后用 /风格启用 挂载生效。"
            )
            if notify:
                await self._notify_analysis(group_id, result)
            return result
        except Exception as exc:
            logger.exception("群 %s 分析失败", group_id)
            if notify:
                await self._notify_analysis(group_id, f"分析失败：群 {group_id} 发生异常（{exc}）。")
            return f"群 {group_id} 分析失败: {exc}"
        finally:
            self._analyzing.discard(group_id)
            if self._store is not None:
                await self._store.set_group_field(group_id, "analyzing", False)

    async def _maybe_auto_merge(self) -> None:
        """配置触发式自动融合：merge.trigger 开启时，把模型 A 与模型 B 融合成新模型。

        插件无法回写配置开关，因此只要开关保持打开，每次保存配置或插件重载都会
        重新执行融合并生成新的模型文件（不会覆盖原模型）；融合完成后建议把开关关回。
        任何失败都只记日志，不影响插件其它功能。
        """

        if self._store is None or self._analyzer is None:
            return
        if not bool(getattr(self.config.merge, "trigger", False)):
            return
        model_a_ref = str(getattr(self.config.merge, "model_a", "") or "").strip()
        model_b_ref = str(getattr(self.config.merge, "model_b", "") or "").strip()
        if not model_a_ref or not model_b_ref:
            logger.warning("配置触发融合：model_a 或 model_b 为空，跳过")
            await self._notify_merge("融合未执行：model_a 或 model_b 为空，请检查配置。")
            return
        new_name = str(getattr(self.config.merge, "new_name", "") or "").strip()
        model_a = self._store.find_model(model_a_ref)
        model_b = self._store.find_model(model_b_ref)
        if model_a is None or model_b is None:
            logger.warning("配置触发融合：找不到模型 %s / %s，请检查模型名或 ID", model_a_ref, model_b_ref)
            await self._notify_merge(f"融合未执行：找不到模型 {model_a_ref} / {model_b_ref}，请检查模型名或 ID。")
            return
        if str(model_a.get("model_id") or "") == str(model_b.get("model_id") or ""):
            logger.warning("配置触发融合：不能融合同一个模型，跳过")
            await self._notify_merge(f"融合未执行：{model_a_ref} 与 {model_b_ref} 是同一个模型，不能融合。")
            return
        if not new_name:
            new_name = f"{model_a.get('name') or '模型A'}×{model_b.get('name') or '模型B'}"
        try:
            result = await self.merge_models(model_a, model_b, new_name)
        except Exception as exc:
            logger.exception("配置触发融合失败: %s", exc)
            return
        if not str(result or "").startswith("融合完成"):
            logger.warning("配置触发融合未成功：%s", result)
            return
        logger.info("配置触发融合完成：%s", result)

    def _merge_options_from_config(self) -> Dict[str, Any]:
        """从配置读取融合优化选项（比例 / 去重强度 / 语句总数上限 / 合并阈值）。"""

        ratio_a = max(0, min(100, int(getattr(self.config.merge, "ratio_a", 50) or 50)))
        dedup_level = str(getattr(self.config.merge, "dedup_level", "") or "标准").strip() or "标准"
        if dedup_level not in ("标准", "激进精简", "不去重"):
            dedup_level = "标准"
        max_total = max(0, int(getattr(self.config.merge, "max_total_statements", 0) or 0))
        scene_threshold = float(getattr(self.config.merge, "scene_merge_threshold", 0.8) or 0.8)
        snapshot_threshold = float(getattr(self.config.merge, "snapshot_merge_threshold", 0.55) or 0.55)
        return {
            "ratio": (ratio_a, 100 - ratio_a),
            "dedup_level": dedup_level,
            "max_total_statements": max_total,
            "scene_merge_threshold": scene_threshold,
            "snapshot_merge_threshold": snapshot_threshold,
        }

    async def merge_models(self, model_a: Dict[str, Any], model_b: Dict[str, Any], new_name: str) -> str:
        """合并两个风格模型（供 /风格合并 与配置触发融合调用），返回结果说明。

        融合过程会分阶段私聊汇报进度（0% 开始、30% LLM 融合中、70% 过滤中、100% 完成），
        未配置 notify_qq 或非好友时静默跳过。
        """

        if self._analyzer is None:
            return "插件尚未就绪。"
        await self._notify_merge(f"开始融合「{model_a.get('name')}」×「{model_b.get('name')}」→「{new_name}」（0%）")
        try:
            await self._notify_merge("LLM 融合中，正在生成合并模型…（30%）")
            merged = await self._analyzer.merge_partials(
                [model_a, model_b],
                name=new_name,
                options=self._merge_options_from_config(),
            )
            if merged is None:
                await self._notify_merge("融合失败：LLM 未返回有效结果。")
                return "合并失败：LLM 未返回有效结果。"
            await self._notify_merge("融合完成，正在执行黑名单过滤…（70%）")

            # 字句黑名单：融合结果中整句包含黑名单字句的语句直接剔除
            phrase_dropped = 0
            for scene in merged.get("scenes") or []:
                statements = scene.get("statements") or []
                kept = [
                    stmt
                    for stmt in statements
                    if not self._text_has_blocked_phrase(str(stmt.get("text") or ""))
                ]
                phrase_dropped += len(statements) - len(kept)
                scene["statements"] = kept

            # 语义黑名单：按相似度过滤（与分析流程一致）
            blacklist = self._store.list_blacklist()
            threshold = float(self.config.blacklist.similarity_threshold)
            semantic_dropped = 0
            if blacklist:
                for scene in merged.get("scenes") or []:
                    kept, dropped = await self._analyzer.filter_statements(
                        scene.get("statements") or [],
                        blacklist,
                        threshold,
                        model_id=str(merged.get("model_id") or ""),
                    )
                    scene["statements"] = kept
                    semantic_dropped += len(dropped)

            merged["model_id"] = self._store.new_model_id()
            merged["group_id"] = model_a.get("group_id") or model_b.get("group_id") or ""
            merged["status"] = "pending"
            merged["created_at"] = time.time()
            merged["sources"] = [model_a.get("model_id"), model_b.get("model_id")]
            merged["dropped_by_blacklist"] = semantic_dropped
            await self._store.save_model(merged)
            result = (
                f"融合完成，已生成模型「{merged.get('name')}」（{len(merged.get('scenes') or [])} 个场景，"
                f"字句黑名单剔除 {phrase_dropped} 条，语义黑名单剔除 {semantic_dropped} 条）。"
                "请审核后用 /风格启用 挂载。"
            )
            await self._notify_merge(result + "（100%）")
            return result
        except Exception as exc:
            logger.exception("模型融合失败")
            await self._notify_merge(f"融合失败：{exc}")
            return f"融合失败: {exc}"

    def _compact_options_from_config(self) -> Dict[str, Any]:
        """从配置读取单模型精简选项（去重强度 / 语句总数上限 / 合并阈值）。"""

        dedup_level = str(getattr(self.config.compact, "dedup_level", "") or "标准").strip() or "标准"
        if dedup_level not in ("标准", "激进精简", "不去重"):
            dedup_level = "标准"
        max_total = max(0, int(getattr(self.config.compact, "max_total_statements", 0) or 0))
        scene_threshold = float(getattr(self.config.compact, "scene_merge_threshold", 0.8) or 0.8)
        snapshot_threshold = float(getattr(self.config.compact, "snapshot_merge_threshold", 0.55) or 0.55)
        return {
            "dedup_level": dedup_level,
            "max_total_statements": max_total,
            "scene_merge_threshold": scene_threshold,
            "snapshot_merge_threshold": snapshot_threshold,
        }

    async def compact_model(self, model: Dict[str, Any], new_name: str = "") -> str:
        """精简单个风格模型（纯本地去重，不调用 LLM），返回结果说明。

        精简后的模型生成全新文件（原模型不动，状态 pending），
        精简过程会分阶段私聊汇报进度（0% 开始、50% 精简中、70% 过滤中、100% 完成）。
        """

        if self._analyzer is None:
            return "插件尚未就绪。"
        name = str(new_name or "").strip() or f"{model.get('name') or '风格模型'}（精简版）"
        await self._notify_merge(
            f"开始精简「{model.get('name')}」→「{name}」（0%）", prefix="【模型精简】"
        )
        try:
            await self._notify_merge("本地去重中，正在合并相似场景与近似语句…（50%）", prefix="【模型精简】")
            compacted, stats = self._analyzer.compact_model(
                model, name=name, options=self._compact_options_from_config()
            )
            await self._notify_merge("精简完成，正在执行黑名单过滤…（70%）", prefix="【模型精简】")

            # 字句黑名单：结果中整句包含黑名单字句的语句直接剔除
            phrase_dropped = 0
            for scene in compacted.get("scenes") or []:
                statements = scene.get("statements") or []
                kept = [
                    stmt
                    for stmt in statements
                    if not self._text_has_blocked_phrase(str(stmt.get("text") or ""))
                ]
                phrase_dropped += len(statements) - len(kept)
                scene["statements"] = kept

            # 语义黑名单：按相似度过滤（与分析流程一致）
            blacklist = self._store.list_blacklist()
            threshold = float(self.config.blacklist.similarity_threshold)
            semantic_dropped = 0
            if blacklist:
                for scene in compacted.get("scenes") or []:
                    kept, dropped = await self._analyzer.filter_statements(
                        scene.get("statements") or [],
                        blacklist,
                        threshold,
                        model_id=str(compacted.get("model_id") or ""),
                    )
                    scene["statements"] = kept
                    semantic_dropped += len(dropped)

            compacted["model_id"] = self._store.new_model_id()
            compacted["group_id"] = str(model.get("group_id") or "")
            compacted["status"] = "pending"
            compacted["created_at"] = time.time()
            compacted["sources"] = [str(model.get("model_id") or "")]
            compacted["dropped_by_blacklist"] = semantic_dropped
            await self._store.save_model(compacted)
            result = (
                f"精简完成，已生成模型「{compacted.get('name')}」（{len(compacted.get('scenes') or [])} 个场景，"
                f"语句去重 {int(stats.get('statement_removed') or 0)} 条，场景合并 {int(stats.get('scene_merged') or 0)} 个，"
                f"字句黑名单剔除 {phrase_dropped} 条，语义黑名单剔除 {semantic_dropped} 条）。"
                "原模型未改动，请审核后用 /风格启用 挂载。"
            )
            await self._notify_merge(result + "（100%）", prefix="【模型精简】")
            return result
        except Exception as exc:
            logger.exception("模型精简失败")
            await self._notify_merge(f"精简失败：{exc}", prefix="【模型精简】")
            return f"精简失败: {exc}"

    async def _maybe_auto_compact(self) -> None:
        """配置触发式自动精简：compact.trigger 开启时，把指定模型精简成新模型。

        插件无法回写配置开关，因此只要开关保持打开，每次保存配置或插件重载都会
        重新执行精简并生成新的模型文件（原模型不动）；精简完成后建议把开关关回。
        任何失败都只记日志，不影响插件其它功能。
        """

        if self._store is None or self._analyzer is None:
            return
        if not bool(getattr(self.config.compact, "trigger", False)):
            return
        model_ref = str(getattr(self.config.compact, "model", "") or "").strip()
        if not model_ref:
            logger.warning("配置触发精简：model 为空，跳过")
            await self._notify_merge("精简未执行：请填写要精简的模型名或 ID。", prefix="【模型精简】")
            return
        new_name = str(getattr(self.config.compact, "new_name", "") or "").strip()
        model = self._store.find_model(model_ref)
        if model is None:
            logger.warning("配置触发精简：找不到模型 %s，请检查模型名或 ID", model_ref)
            await self._notify_merge(
                f"精简未执行：找不到模型 {model_ref}，请检查模型名或 ID。", prefix="【模型精简】"
            )
            return
        try:
            result = await self.compact_model(model, new_name)
        except Exception as exc:
            logger.exception("配置触发精简失败: %s", exc)
            return
        if not str(result or "").startswith("精简完成"):
            logger.warning("配置触发精简未成功：%s", result)
            return
        logger.info("配置触发精简完成：%s", result)

    # ------------------------------------------------------------------
    # 命令：群管理
    # ------------------------------------------------------------------

    @Command("style_help", description="聊天风格转译帮助", pattern=r"^/风格帮助$")
    async def cmd_help(self, **kwargs: Any) -> tuple[bool, str, bool]:
        """返回使用说明。"""

        return await self._reply(kwargs, HELP_TEXT)

    @Command(
        "style_bind_group",
        description="绑定群开始收集对话",
        pattern=r"^/风格绑定群\s+(?P<group_id>\d+)$",
    )
    async def cmd_bind_group(self, **kwargs: Any) -> tuple[bool, str, bool]:
        """绑定一个 QQ 群，开始自动收集群友对话。"""

        if not self._is_owner(kwargs):
            return await self._deny(kwargs)
        if self._store is None:
            return await self._reply(kwargs, "插件尚未就绪。")
        group_id = str((kwargs.get("matched_groups") or {}).get("group_id") or "")
        if not group_id:
            return await self._reply(kwargs, "群号格式不正确。")
        await self._store.bind_group(group_id)
        stream_id = str(kwargs.get("stream_id") or "")
        if stream_id:
            self._remember_session_mapping(stream_id, group_id)
        auto_count = self.config.collection.auto_analyze_count
        daily_time = str(self.config.collection.daily_time or "").strip()
        trigger_desc = f"收集满 {auto_count} 条自动分析" if auto_count > 0 else "不按条数触发"
        if daily_time:
            trigger_desc += f"，每日 {daily_time} 触发"
        return await self._reply(kwargs, f"已绑定群 {group_id}，开始自动收集对话（{trigger_desc}）。")

    @Command(
        "style_unbind_group",
        description="解绑群并删除收集的对话",
        pattern=r"^/风格解绑群\s+(?P<group_id>\d+)$",
    )
    async def cmd_unbind_group(self, **kwargs: Any) -> tuple[bool, str, bool]:
        """解绑群。"""

        if not self._is_owner(kwargs):
            return await self._deny(kwargs)
        if self._store is None:
            return await self._reply(kwargs, "插件尚未就绪。")
        group_id = str((kwargs.get("matched_groups") or {}).get("group_id") or "")
        if await self._store.unbind_group(group_id):
            return await self._reply(kwargs, f"已解绑群 {group_id}，收集的对话已删除。")
        return await self._reply(kwargs, f"群 {group_id} 未绑定。")

    @Command("style_status", description="查看收集与任务状态", pattern=r"^/风格状态$")
    async def cmd_status(self, **kwargs: Any) -> tuple[bool, str, bool]:
        """查看各绑定群的收集进度。"""

        if self._store is None:
            return await self._reply(kwargs, "插件尚未就绪。")
        lines = ["【绑定群收集状态】"]
        groups = self._store.list_groups()
        if not groups:
            lines.append("尚未绑定任何群。使用 /风格绑定群 <群号> 开始收集。")
        for group_id, group in groups.items():
            count = self._store.message_count(group_id)
            name = str(group.get("group_name") or "")
            active = group.get("active_models") or []
            analyzing = "（分析中）" if group_id in self._analyzing else ""
            last = group.get("last_analyzed_at") or 0
            last_text = datetime.fromtimestamp(last).strftime("%m-%d %H:%M") if last else "从未"
            backfilled = group.get("last_backfilled_at") or 0
            backfill_text = datetime.fromtimestamp(backfilled).strftime("%m-%d %H:%M") if backfilled else "从未"
            lines.append(
                f"- {name} ({group_id})：已收集 {count} 条，挂载模型 {len(active)} 个，"
                f"上次分析 {last_text}，上次历史回填 {backfill_text}{analyzing}"
            )
        return await self._reply(kwargs, "\n".join(lines))

    @Command(
        "style_generate",
        description="立即对已收集的对话执行转译分析",
        pattern=r"^/风格生成(?:\s+(?P<group_id>\d+))?$",
    )
    async def cmd_generate(self, **kwargs: Any) -> tuple[bool, str, bool]:
        """手动触发分析：后台执行并立即返回（进度与结果私聊通知，避免命令超时）。

        未指定群号时在群内分析当前群，私聊/无群上下文则分析全部绑定群。
        """

        if not self._is_owner(kwargs):
            return await self._deny(kwargs)
        if self._store is None:
            return await self._reply(kwargs, "插件尚未就绪。")
        group_id = str((kwargs.get("matched_groups") or {}).get("group_id") or "").strip()
        if not group_id:
            group_id = self._extract_group_id(kwargs.get("message") or {})
        if not group_id:
            group_ids = [str(gid) for gid in self._store.list_groups().keys()]
            if not group_ids:
                return await self._reply(kwargs, "尚未绑定任何群，无法分析。")
            busy = [gid for gid in group_ids if gid in self._analyzing]
            if busy:
                return await self._reply(kwargs, f"以下群正在分析中，请稍后再试：{'、'.join(busy)}")
            for gid in group_ids:
                asyncio.get_running_loop().create_task(
                    self._run_analysis(gid, reason="手动触发（全部群）", notify=True)
                )
            return await self._reply(kwargs, f"已开始对 {len(group_ids)} 个绑定群执行转译分析（后台执行）。进度与结果会私聊通知。")
        if group_id in self._analyzing:
            return await self._reply(kwargs, f"群 {group_id} 正在分析中，请稍后再试。")
        asyncio.get_running_loop().create_task(
            self._run_analysis(group_id, reason="手动触发", notify=True)
        )
        return await self._reply(kwargs, f"已开始分析群 {group_id}（后台执行）。进度与结果会私聊通知。")

    @Command(
        "style_backfill",
        description="从 NapCat 拉取群历史消息回填收集",
        pattern=r"^/风格拉取历史(?:\s+(?P<group_id>\d+))?(?:\s+(?P<count>\d+))?$",
    )
    async def cmd_backfill(self, **kwargs: Any) -> tuple[bool, str, bool]:
        """手动触发历史回填：后台执行并立即返回（结果私聊通知，避免命令超时）。

        拉取该群最近 N 条历史消息补进收集池（N 默认取配置「历史回填条数」）；
        未指定群号时在群内回填当前群，私聊/无群上下文则回填全部绑定群；
        与实时收集一致的屏蔽与字句黑名单过滤、按消息 ID 去重；
        回填后收集条数达到自动分析阈值会自动触发后台分析。
        """

        if not self._is_owner(kwargs):
            return await self._deny(kwargs)
        if self._store is None:
            return await self._reply(kwargs, "插件尚未就绪。")
        groups = kwargs.get("matched_groups") or {}
        group_id = str(groups.get("group_id") or "").strip()
        count = str(groups.get("count") or "").strip()
        if not group_id:
            group_id = self._extract_group_id(kwargs.get("message") or {})
        count = int(count) if count.isdigit() else 0
        if not group_id:
            group_ids = [str(gid) for gid in self._store.list_groups().keys()]
            if not group_ids:
                return await self._reply(kwargs, "尚未绑定任何群，无法回填。")
            for gid in group_ids:
                asyncio.get_running_loop().create_task(
                    self._run_backfill(gid, count=count, notify=True)
                )
            return await self._reply(kwargs, f"已开始对 {len(group_ids)} 个绑定群执行历史回填（后台执行）。结果会私聊通知。")
        asyncio.get_running_loop().create_task(
            self._run_backfill(group_id, count=count, notify=True)
        )
        return await self._reply(kwargs, f"已开始回填群 {group_id}（后台执行）。结果会私聊通知。")

    # ------------------------------------------------------------------
    # 命令：模型管理
    # ------------------------------------------------------------------

    @Command("style_list", description="列出全部风格模型", pattern=r"^/风格列表$")
    async def cmd_list(self, **kwargs: Any) -> tuple[bool, str, bool]:
        """列出全部模型。"""

        if self._store is None:
            return await self._reply(kwargs, "插件尚未就绪。")
        models = self._store.list_models()
        if not models:
            return await self._reply(kwargs, "还没有任何风格模型。先绑定群收集对话，再用 /风格生成 分析。")
        lines = [f"【风格模型列表】模型目录：{self._store.model_dir}"]
        for model in models:
            scenes = len(model.get("scenes") or [])
            statements = sum(len(s.get("statements") or []) for s in model.get("scenes") or [])
            status = model.get("status") or "pending"
            global_flag = "（全局通用）" if bool(model.get("global_apply")) else ""
            lines.append(
                f"- {model.get('name')}{global_flag}（{model.get('model_id')}）：{scenes} 场景 / {statements} 语句，状态：{status}"
            )
        return await self._reply(kwargs, "\n".join(lines))

    @Command(
        "style_detail",
        description="查看模型摘要",
        pattern=r"^/风格详情\s+(?P<model_name>\S+)$",
    )
    async def cmd_detail(self, **kwargs: Any) -> tuple[bool, str, bool]:
        """查看模型摘要。"""

        if self._store is None:
            return await self._reply(kwargs, "插件尚未就绪。")
        model = self._store.find_model(str((kwargs.get("matched_groups") or {}).get("model_name") or ""))
        if model is None:
            return await self._reply(kwargs, "未找到该模型。")
        lines = [
            f"模型：{model.get('name')}",
            f"ID：{model.get('model_id')}",
            f"状态：{model.get('status')}",
            f"全局通用：{'开' if bool(model.get('global_apply')) else '关'}",
        ]
        snapshots = [str(s).strip() for s in (model.get("style_snapshots") or []) if str(s or "").strip()]
        if len(snapshots) > 1:
            lines.append(f"整体风格（{len(snapshots)} 个快照，不合并）：")
            for index, snapshot in enumerate(snapshots, start=1):
                lines.append(f"- 快照{index}：{snapshot}")
        else:
            lines.append(f"整体风格：{model.get('overall_style') or '（无）'}")
        members = model.get("members") or []
        if members:
            lines.append("群友档案：")
            for member in members[:8]:
                user_id = str(member.get("user_id") or "").strip()
                label = f"QQ {user_id} " if user_id else ""
                personality = str(member.get("personality") or "").strip()
                style = str(member.get("style") or "").strip()
                detail = "；".join(part for part in [f"性格：{personality}", f"风格：{style}"] if part)
                lines.append(f"- {label}{member.get('name')}：{detail or '（无）'}")
        scenes = model.get("scenes") or []
        if scenes:
            lines.append("场景：")
            for index, scene in enumerate(scenes, start=1):
                statements = len(scene.get("statements") or [])
                note = str(scene.get("note") or "").strip()
                note_text = f" - {note}" if note else ""
                lines.append(f"  {index}. {scene.get('name')}（{statements} 条语句）: {scene.get('description') or ''}{note_text}")
        lines.append("提示：用 /风格场景 <模型名> <序号> 查看场景语句，/风格启用 <模型名> 审核并挂载。")
        return await self._reply(kwargs, "\n".join(lines))

    @Command(
        "style_scene",
        description="查看模型场景语句",
        pattern=r"^/风格场景\s+(?P<model_name>\S+)(?:\s+(?P<scene_index>\d+))?$",
    )
    async def cmd_scene(self, **kwargs: Any) -> tuple[bool, str, bool]:
        """查看模型指定场景的语句明细。"""

        if self._store is None:
            return await self._reply(kwargs, "插件尚未就绪。")
        groups = kwargs.get("matched_groups") or {}
        model = self._store.find_model(str(groups.get("model_name") or ""))
        if model is None:
            return await self._reply(kwargs, "未找到该模型。")
        scenes = model.get("scenes") or []
        if not scenes:
            return await self._reply(kwargs, "该模型没有任何场景。")
        raw_index = str(groups.get("scene_index") or "").strip()
        if raw_index:
            try:
                index = int(raw_index) - 1
                scenes = [scenes[index]]
            except (ValueError, IndexError):
                return await self._reply(kwargs, "场景序号无效。")
        lines = [f"模型：{model.get('name')}（{model.get('model_id')}）"]
        for scene in scenes:
            lines.append(f"\n【场景：{scene.get('name')}】")
            if scene.get("style"):
                lines.append(f"风格：{scene.get('style')}")
            if scene.get("note"):
                lines.append(f"场景备注：{scene.get('note')}")
            statements = scene.get("statements") or []
            if not statements:
                lines.append("（该场景暂无语句）")
            for statement in statements:
                target = statement.get("target") or "大家"
                meaning = statement.get("meaning") or ""
                ai_note = statement.get("ai_note") or ""
                lines.append(f"- {statement.get('id')} {statement.get('speaker_name')}对{target}：“{statement.get('text')}”")
                if meaning:
                    lines.append(f"  意思：{meaning}")
                if ai_note:
                    lines.append(f"  AI理解：{ai_note}")
        return await self._reply(kwargs, "\n".join(lines))

    @Command(
        "style_enable",
        description="审核通过并挂载模型到群",
        pattern=r"^/风格启用\s+(?P<model_name>\S+)(?:\s+(?P<group_id>\d+))?$",
    )
    async def cmd_enable(self, **kwargs: Any) -> tuple[bool, str, bool]:
        """审核通过模型并挂载到指定群（默认当前群）。"""

        if not self._is_owner(kwargs):
            return await self._deny(kwargs)
        if self._store is None:
            return await self._reply(kwargs, "插件尚未就绪。")
        groups = kwargs.get("matched_groups") or {}
        model = self._store.find_model(str(groups.get("model_name") or ""))
        if model is None:
            return await self._reply(kwargs, "未找到该模型。")
        group_id = str(groups.get("group_id") or "").strip()
        if not group_id:
            group_id = self._extract_group_id(kwargs.get("message") or {})
        if not group_id:
            return await self._reply(kwargs, "请在群内执行，或指定群号：/风格启用 <模型名> <群号>")
        if not self._store.is_bound(group_id):
            return await self._reply(kwargs, f"群 {group_id} 未绑定，请先 /风格绑定群 {group_id}。")
        model_file = self._store.find_model_file(str(groups.get("model_name") or ""))
        if model_file is None:
            return await self._reply(kwargs, "未找到该模型。")
        model = self._store.get_model(model_file)
        if model.get("status") != "approved":
            model["status"] = "approved"
            await self._store.save_model(model)
        group = self._store.get_group(group_id)
        active = group.get("active_models") or []
        if isinstance(active, list) and model_file not in active:
            active.append(model_file)
            await self._store.set_group_field(group_id, "active_models", active)
        return await self._reply(kwargs, f"模型「{model.get('name')}」已审核通过并挂载到群 {group_id}，AI 生成回复后将按此风格轻度微调措辞（只改几个词，原意不变）。")

    @Command(
        "style_disable",
        description="从群上摘除模型",
        pattern=r"^/风格停用\s+(?P<model_name>\S+)(?:\s+(?P<group_id>\d+))?$",
    )
    async def cmd_disable(self, **kwargs: Any) -> tuple[bool, str, bool]:
        """从群上摘除模型。"""

        if not self._is_owner(kwargs):
            return await self._deny(kwargs)
        if self._store is None:
            return await self._reply(kwargs, "插件尚未就绪。")
        groups = kwargs.get("matched_groups") or {}
        model = self._store.find_model(str(groups.get("model_name") or ""))
        if model is None:
            return await self._reply(kwargs, "未找到该模型。")
        group_id = str(groups.get("group_id") or "").strip()
        if not group_id:
            group_id = self._extract_group_id(kwargs.get("message") or {})
        if not group_id:
            return await self._reply(kwargs, "请在群内执行，或指定群号：/风格停用 <模型名> <群号>")
        group = self._store.get_group(group_id)
        if group is None:
            return await self._reply(kwargs, f"群 {group_id} 未绑定。")
        active = group.get("active_models") or []
        model_file = self._store.find_model_file(str(groups.get("model_name") or ""))
        if model_file is None:
            return await self._reply(kwargs, "未找到该模型。")
        if isinstance(active, list) and model_file in active:
            active.remove(model_file)
            await self._store.set_group_field(group_id, "active_models", active)
            return await self._reply(kwargs, f"模型「{model.get('name')}」已从群 {group_id} 摘除。")
        return await self._reply(kwargs, f"模型「{model.get('name')}」未挂载到群 {group_id}。")

    @Command(
        "style_global_enable",
        description="开启模型全局通用（对所有群生效）",
        pattern=r"^/风格全局启用\s+(?P<model_name>\S+)$",
    )
    async def cmd_global_enable(self, **kwargs: Any) -> tuple[bool, str, bool]:
        """开启模型全局通用：该模型对所有群聊（含未绑定群）都生效。"""

        if not self._is_owner(kwargs):
            return await self._deny(kwargs)
        if self._store is None:
            return await self._reply(kwargs, "插件尚未就绪。")
        model = self._store.find_model(str((kwargs.get("matched_groups") or {}).get("model_name") or ""))
        if model is None:
            return await self._reply(kwargs, "未找到该模型。")
        if model.get("status") != "approved":
            model["status"] = "approved"
        model["global_apply"] = True
        await self._store.save_model(model)
        await self._refresh_global_models()
        return await self._reply(kwargs, f"模型「{model.get('name')}」已开启全局通用，所有群聊都会使用该风格。")

    @Command(
        "style_global_disable",
        description="关闭模型全局通用",
        pattern=r"^/风格全局停用\s+(?P<model_name>\S+)$",
    )
    async def cmd_global_disable(self, **kwargs: Any) -> tuple[bool, str, bool]:
        """关闭模型全局通用。"""

        if not self._is_owner(kwargs):
            return await self._deny(kwargs)
        if self._store is None:
            return await self._reply(kwargs, "插件尚未就绪。")
        model = self._store.find_model(str((kwargs.get("matched_groups") or {}).get("model_name") or ""))
        if model is None:
            return await self._reply(kwargs, "未找到该模型。")
        model["global_apply"] = False
        await self._store.save_model(model)
        await self._refresh_global_models()
        return await self._reply(kwargs, f"模型「{model.get('name')}」已关闭全局通用。")

    @Command(
        "style_merge",
        description="融合两个风格模型",
        pattern=r"^/风格合并\s+(?P<model_a>\S+)\s+(?P<model_b>\S+)(?:\s+(?P<new_name>\S+))?$",
    )
    async def cmd_merge(self, **kwargs: Any) -> tuple[bool, str, bool]:
        """融合两个模型为一份新模型。"""

        if not self._is_owner(kwargs):
            return await self._deny(kwargs)
        if self._store is None:
            return await self._reply(kwargs, "插件尚未就绪。")
        groups = kwargs.get("matched_groups") or {}
        model_a = self._store.find_model(str(groups.get("model_a") or ""))
        model_b = self._store.find_model(str(groups.get("model_b") or ""))
        if model_a is None or model_b is None:
            return await self._reply(kwargs, "未找到要融合的模型，请检查模型名。")
        if model_a.get("model_id") == model_b.get("model_id"):
            return await self._reply(kwargs, "不能融合同一个模型。")
        new_name = str(groups.get("new_name") or "").strip() or f"{model_a.get('name')}×{model_b.get('name')}"
        result = await self.merge_models(model_a, model_b, new_name)
        return await self._reply(kwargs, result)

    @Command(
        "style_remove_statement",
        description="删除模型中的语句",
        pattern=r"^/风格删除语句\s+(?P<model_name>\S+)\s+(?P<statement_id>\S+)$",
    )
    async def cmd_remove_statement(self, **kwargs: Any) -> tuple[bool, str, bool]:
        """删除模型中不符合场景的语句。"""

        if not self._is_owner(kwargs):
            return await self._deny(kwargs)
        if self._store is None:
            return await self._reply(kwargs, "插件尚未就绪。")
        groups = kwargs.get("matched_groups") or {}
        model = self._store.find_model(str(groups.get("model_name") or ""))
        if model is None:
            return await self._reply(kwargs, "未找到该模型。")
        statement_id = str(groups.get("statement_id") or "").strip()
        removed = False
        for scene in model.get("scenes") or []:
            statements = scene.get("statements") or []
            remaining = [s for s in statements if str(s.get("id") or "") != statement_id]
            if len(remaining) != len(statements):
                scene["statements"] = remaining
                removed = True
        if not removed:
            return await self._reply(kwargs, f"模型「{model.get('name')}」中未找到语句 {statement_id}。")
        await self._store.save_model(model)
        return await self._reply(kwargs, f"已删除语句 {statement_id}。")

    @Command(
        "style_remove_scene",
        description="删除模型中的场景",
        pattern=r"^/风格删除场景\s+(?P<model_name>\S+)\s+(?P<scene_ref>\S+)$",
    )
    async def cmd_remove_scene(self, **kwargs: Any) -> tuple[bool, str, bool]:
        """删除模型中的整个场景。"""

        if not self._is_owner(kwargs):
            return await self._deny(kwargs)
        if self._store is None:
            return await self._reply(kwargs, "插件尚未就绪。")
        groups = kwargs.get("matched_groups") or {}
        model = self._store.find_model(str(groups.get("model_name") or ""))
        if model is None:
            return await self._reply(kwargs, "未找到该模型。")
        scene_ref = str(groups.get("scene_ref") or "").strip()
        scenes = model.get("scenes") or []
        removed_scene = None
        if scene_ref.isdigit():
            index = int(scene_ref) - 1
            if 0 <= index < len(scenes):
                removed_scene = scenes.pop(index)
        else:
            remaining = [s for s in scenes if str(s.get("name") or "") != scene_ref]
            if len(remaining) != len(scenes):
                removed_scene = next(s for s in scenes if str(s.get("name") or "") == scene_ref)
                model["scenes"] = remaining
        if removed_scene is None:
            return await self._reply(kwargs, f"模型「{model.get('name')}」中未找到该场景。")
        await self._store.save_model(model)
        return await self._reply(kwargs, f"已删除场景「{removed_scene.get('name')}」。")

    @Command(
        "style_rename",
        description="重命名模型",
        pattern=r"^/风格重命名\s+(?P<model_name>\S+)\s+(?P<new_name>\S+)$",
    )
    async def cmd_rename(self, **kwargs: Any) -> tuple[bool, str, bool]:
        """重命名模型。"""

        if not self._is_owner(kwargs):
            return await self._deny(kwargs)
        if self._store is None:
            return await self._reply(kwargs, "插件尚未就绪。")
        groups = kwargs.get("matched_groups") or {}
        model = self._store.find_model(str(groups.get("model_name") or ""))
        if model is None:
            return await self._reply(kwargs, "未找到该模型。")
        new_name = str(groups.get("new_name") or "").strip()
        model["name"] = new_name
        await self._store.save_model(model)
        return await self._reply(kwargs, f"模型已重命名为「{new_name}」。")

    @Command(
        "style_delete_model",
        description="删除模型",
        pattern=r"^/风格删除模型\s+(?P<model_name>\S+)$",
    )
    async def cmd_delete_model(self, **kwargs: Any) -> tuple[bool, str, bool]:
        """删除模型。"""

        if not self._is_owner(kwargs):
            return await self._deny(kwargs)
        if self._store is None:
            return await self._reply(kwargs, "插件尚未就绪。")
        model = self._store.find_model(str((kwargs.get("matched_groups") or {}).get("model_name") or ""))
        if model is None:
            return await self._reply(kwargs, "未找到该模型。")
        await self._store.delete_model(str(model.get("model_id") or ""))
        return await self._reply(kwargs, f"已删除模型「{model.get('name')}」。")

    # ------------------------------------------------------------------
    # 命令：黑名单
    # ------------------------------------------------------------------

    @Command(
        "blacklist_add",
        description="添加通用黑名单语句",
        pattern=r"^/黑名单添加\s+(?P<text>.+)$",
    )
    async def cmd_blacklist_add(self, **kwargs: Any) -> tuple[bool, str, bool]:
        """添加黑名单语句（通用拉黑：按相似度对所有模型生效）。"""

        if not self._is_owner(kwargs):
            return await self._deny(kwargs)
        if self._store is None:
            return await self._reply(kwargs, "插件尚未就绪。")
        text = str((kwargs.get("matched_groups") or {}).get("text") or "").strip()
        if not text:
            return await self._reply(kwargs, "语句不能为空。")
        item = await self._store.add_blacklist(text, apply_all=True)
        return await self._reply(kwargs, (
            f"已添加通用黑名单：{text}（ID: {item.get('id')}）。"
            "生成模型时，含义与该语句高度相似的语句会在所有模型中被自动过滤。"
        ))

    @Command(
        "blacklist_add_model",
        description="向指定模型添加黑名单语句",
        pattern=r"^/黑名单添加模型\s+(?P<model_name>\S+)\s+(?P<text>.+)$",
    )
    async def cmd_blacklist_add_model(self, **kwargs: Any) -> tuple[bool, str, bool]:
        """向指定模型添加黑名单语句（只在该模型内按相似度过滤）。"""

        if not self._is_owner(kwargs):
            return await self._deny(kwargs)
        if self._store is None:
            return await self._reply(kwargs, "插件尚未就绪。")
        groups = kwargs.get("matched_groups") or {}
        model = self._store.find_model(str(groups.get("model_name") or ""))
        if model is None:
            return await self._reply(kwargs, "未找到该模型。")
        text = str(groups.get("text") or "").strip()
        if not text:
            return await self._reply(kwargs, "语句不能为空。")
        item = await self._store.add_blacklist(
            text,
            model_id=str(model.get("model_id") or ""),
            model_name=str(model.get("name") or ""),
            apply_all=False,
        )
        return await self._reply(kwargs, (
            f"已向模型「{model.get('name')}」添加黑名单：{text}（ID: {item.get('id')}）。"
            "生成该模型时，含义高度相似的语句会被自动过滤。"
        ))

    @Command(
        "blacklist_remove",
        description="删除黑名单语句",
        pattern=r"^/黑名单删除\s+(?P<item_id>\S+)$",
    )
    async def cmd_blacklist_remove(self, **kwargs: Any) -> tuple[bool, str, bool]:
        """删除黑名单条目。"""

        if not self._is_owner(kwargs):
            return await self._deny(kwargs)
        if self._store is None:
            return await self._reply(kwargs, "插件尚未就绪。")
        item_id = str((kwargs.get("matched_groups") or {}).get("item_id") or "")
        if await self._store.remove_blacklist(item_id):
            return await self._reply(kwargs, f"已删除黑名单条目 {item_id}。")
        return await self._reply(kwargs, f"未找到黑名单条目 {item_id}。")

    @Command("blacklist_list", description="查看黑名单", pattern=r"^/黑名单列表$")
    async def cmd_blacklist_list(self, **kwargs: Any) -> tuple[bool, str, bool]:
        """列出黑名单条目。"""

        if self._store is None:
            return await self._reply(kwargs, "插件尚未就绪。")
        items = self._store.list_blacklist()
        if not items:
            return await self._reply(kwargs, "黑名单为空。可在 WebUI 插件配置 → 黑名单 中编辑，或用 /黑名单添加 命令。")
        lines = ["【黑名单】"]
        for item in items:
            scope = "通用" if bool(item.get("apply_all")) else f"模型「{item.get('model_name') or item.get('model_id') or '未知'}」"
            text = str(item.get("text") or "")
            meaning = str(item.get("meaning") or "").strip()
            detail = f"{text}（{meaning}）" if meaning else text
            lines.append(f"- {item.get('id')} [{scope}]：{detail}")
        lines.append("提示：可在 WebUI 插件配置 → 黑名单 中添加带含义描述、限定模型的黑名单条目。")
        return await self._reply(kwargs, "\n".join(lines))

    @Command("blacklist_clear", description="清空黑名单", pattern=r"^/黑名单清空$")
    async def cmd_blacklist_clear(self, **kwargs: Any) -> tuple[bool, str, bool]:
        """清空黑名单。"""

        if not self._is_owner(kwargs):
            return await self._deny(kwargs)
        if self._store is None:
            return await self._reply(kwargs, "插件尚未就绪。")
        await self._store.clear_blacklist()
        return await self._reply(kwargs, "黑名单已清空。")


def create_plugin() -> ChatStyleTranslatorPlugin:
    """创建聊天风格转译插件实例。"""

    return ChatStyleTranslatorPlugin()
