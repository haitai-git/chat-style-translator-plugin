from __future__ import annotations

import re
from functools import wraps

from maibot_sdk import Command

_ARGUMENT = r'(?:"[^"\r\n]+"|“[^”\r\n]+”|「[^」\r\n]+」|[^"\s“”「」]+)'
_QUOTE_PAIRS = {('"', '"'), ('“', '”'), ('「', '」')}


def _unquote(value: str) -> str:
    if (value[0], value[-1]) in _QUOTE_PAIRS:
        return value[1:-1].strip()
    return value


def style_command(
    name: str, trigger: str, *,
    description: str, syntax: str = "", usage: str = "",
    admin: bool = True, aliases: tuple[str, ...] = (),
):
    words = "|".join(re.escape(word) for word in (trigger, *aliases))
    route = rf"^\s*[/／]\s*(?:{words})(?=\s|$)(?P<style_args>[\s\S]*)$"
    arguments = re.compile(syntax.replace("{arg}", _ARGUMENT))
    instructions = f"用法：/{trigger}" + (f" {usage}" if usage else "")

    def decorate(handler):
        @wraps(handler)
        async def invoke(self, **kwargs):
            if admin and not self._is_owner(kwargs):
                return await self._deny(kwargs)
            provided = kwargs.get("matched_groups") or {}
            if "style_args" in provided:
                raw = str(provided["style_args"] or "").strip()
                parsed = arguments.fullmatch(raw)
                if parsed is None:
                    return await self._reply(kwargs, instructions)
                values = {key: value for key, value in parsed.groupdict().items() if value is not None}
                for key, value in values.items():
                    if key != "text":
                        values[key] = _unquote(value)
                        if not values[key]:
                            return await self._reply(kwargs, instructions)
                kwargs["matched_groups"] = values
            return await handler(self, **kwargs)

        return Command(name, description=description, pattern=route)(invoke)

    return decorate
