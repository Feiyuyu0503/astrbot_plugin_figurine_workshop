import asyncio
import base64
import io
import random
import re
from datetime import datetime
from pathlib import Path

import aiohttp
from PIL import Image as PILImage

import astrbot.core.message.components as Comp
from astrbot.api import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import Image
from astrbot.core.platform.astr_message_event import AstrMessageEvent


class ImageWorkflow:
    def __init__(self):
        self.session = aiohttp.ClientSession()

    async def _download_image(self, url: str) -> bytes | None:
        try:
            async with self.session.get(url) as resp:
                resp.raise_for_status()
                return await resp.read()
        except Exception as e:
            logger.error(f"图片下载失败: {e}")
            return None

    async def _get_avatar(self, user_id: str) -> bytes | None:
        if not user_id.isdigit():
            user_id = "".join(random.choices("0123456789", k=9))
        avatar_url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
        try:
            async with self.session.get(avatar_url, timeout=10) as resp:
                resp.raise_for_status()
                return await resp.read()
        except Exception as e:
            logger.error(f"下载头像失败: {e}")
            return None

    def _extract_first_frame_sync(self, raw: bytes) -> bytes:
        """
        使用PIL库处理图片数据。如果是GIF，则提取第一帧并转为PNG。
        """
        img_io = io.BytesIO(raw)
        img = PILImage.open(img_io)
        if img.format != "GIF":
            return raw
        logger.info("检测到GIF, 将抽取 GIF 的第一帧来生图")
        first_frame = img.convert("RGBA")
        out_io = io.BytesIO()
        first_frame.save(out_io, format="PNG")
        return out_io.getvalue()

    async def _load_bytes(self, src: str) -> bytes | None:
        raw: bytes | None = None
        loop = asyncio.get_running_loop()

        if Path(src).is_file():
            raw = await loop.run_in_executor(None, Path(src).read_bytes)
        elif src.startswith("http"):
            raw = await self._download_image(src)
        elif src.startswith("base64://"):
            raw = await loop.run_in_executor(None, base64.b64decode, src[9:])

        if not raw:
            return None
        return await loop.run_in_executor(None, self._extract_first_frame_sync, raw)

    async def get_first_image(self, event: AstrMessageEvent) -> bytes | None:
        for s in event.message_obj.message:
            if isinstance(s, Comp.Reply) and s.chain:
                for seg in s.chain:
                    if isinstance(seg, Comp.Image):
                        if seg.url and (img := await self._load_bytes(seg.url)):
                            return img
                        if seg.file and (img := await self._load_bytes(seg.file)):
                            return img
        for seg in event.message_obj.message:
            if isinstance(seg, Comp.Image):
                if seg.url and (img := await self._load_bytes(seg.url)):
                    return img
                if seg.file and (img := await self._load_bytes(seg.file)):
                    return img
            elif isinstance(seg, Comp.At):
                if avatar := await self._get_avatar(str(seg.qq)):
                    return avatar
        return await self._get_avatar(event.get_sender_id())

    async def terminate(self):
        if self.session and not self.session.closed:
            await self.session.close()


@register(
    "astrbot_plugin_figurine_workshop",
    "长安某",
    "使用 Gemini API 将图片手办化",
    "1.0.0",
)
class LMArenaPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        self.save_image = config.get("save_image", False)
        self.plugin_data_dir = StarTools.get_data_dir(
            "astrbot_plugin_figurine_workshop"
        )
        self.api_keys = self.conf.get("gemini_api_keys", [])
        self.current_key_index = 0
        self.api_base_url = self.conf.get(
            "api_base_url", "https://generativelanguage.googleapis.com"
        )
        self.figurine_style = self.conf.get("figurine_style", "deluxe_box")
        if not self.api_keys:
            logger.error("LMArenaPlugin: 未配置任何 Gemini API 密钥")

    async def initialize(self):
        self.iwf = ImageWorkflow()

    @filter.regex(r"^(手办化)", priority=3)
    async def on_nano(self, event: AstrMessageEvent):
        img_bytes = await self.iwf.get_first_image(event)
        if not img_bytes:
            yield event.plain_result("缺少图片参数（可以发送图片或@用户）")
            return

        user_prompt = re.sub(
            r"^(手办化)\s*", "", event.message_obj.message_str, count=1
        ).strip()
        yield event.plain_result(
            f"正在生成 [{self.figurine_style}] 风格手办，请稍等..."
        )
        res = await self._generate_figurine_with_gemini(img_bytes, user_prompt)

        if isinstance(res, bytes):
            yield event.chain_result([Image.fromBytes(res)])
            if self.save_image:
                save_path = (
                    self.plugin_data_dir
                    / f"gemini_{self.figurine_style}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
                )

                def write_file():
                    with save_path.open("wb") as f:
                        f.write(res)

                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, write_file)

        elif isinstance(res, str):
            yield event.plain_result(f"生成失败: {res}")
        else:
            yield event.plain_result("生成失败，发生未知错误。")

    async def _generate_figurine_with_gemini(
        self, image_bytes: bytes, user_prompt: str
    ) -> bytes | str | None:
        prompts_config = self.conf.get("prompts", {})
        base_prompt = prompts_config.get(self.figurine_style)

        if not base_prompt:
            error_msg = (
                f"配置错误：未能在配置文件中找到名为 '{self.figurine_style}' 的提示词。"
            )
            logger.error(error_msg)
            return error_msg

        final_prompt = (
            f"{base_prompt}\n\nAdditional user requirements from user: {user_prompt}"
            if user_prompt
            else base_prompt
        )
        logger.info(f"Gemini 手办化 Prompt ({self.figurine_style}): {final_prompt}")

        async def edit_operation(api_key):
            model_name = self.conf.get("model_name", "gemini-2.5-flash-image-preview")
            request_format = self.conf.get("request_format", "gemini")
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            if request_format == "gemini":
                payload = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {"text": final_prompt},
                                {
                                    "inlineData": {
                                        "mimeType": "image/png",
                                        "data": image_base64,
                                    }
                                },
                            ],
                        }
                    ],
                    "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
                }
                return await self._send_image_request(model_name, payload, api_key)
            elif request_format == "openai":
                payload = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": final_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                }
                return await self._send_image_request_openai(model_name, payload, api_key)

        image_data = await self._with_retry(edit_operation)
        if not image_data:
            return "所有API密钥均尝试失败"
        return image_data

    def _get_current_key(self):
        if not self.api_keys:
            return None
        return self.api_keys[self.current_key_index]

    def _switch_key(self):
        if not self.api_keys:
            return
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        logger.info(f"切换到下一个 Gemini API 密钥（索引：{self.current_key_index}）")

    async def _send_image_request(self, model_name, payload, api_key):
        base_url = self.api_base_url.strip().removesuffix("/")
        endpoint = (
            f"{base_url}/v1beta/models/{model_name}:generateContent?key={api_key}"
        )
        headers = {"Content-Type": "application/json"}

        async with self.iwf.session.post(
            url=endpoint, json=payload, headers=headers
        ) as response:
            if response.status != 200:
                response_text = await response.text()
                logger.error(
                    f"API请求失败: HTTP {response.status}, 响应: {response_text}"
                )
                response.raise_for_status()
            data = await response.json()

        if (
            "candidates" in data
            and data["candidates"]
            and "content" in data["candidates"][0]
            and "parts" in data["candidates"][0]["content"]
        ):
            for part in data["candidates"][0]["content"]["parts"]:
                if "inlineData" in part and "data" in part["inlineData"]:
                    return base64.b64decode(part["inlineData"]["data"])

        raise Exception("操作成功，但未在响应中获取到图片数据")

    async def _send_image_request_openai(self, model_name, payload, api_key):
        base_url = self.api_base_url.strip().removesuffix("/")
        endpoint = (
            f"{base_url}/v1/chat/completions"
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
            }
        async with self.iwf.session.post(
            url=endpoint, json=payload, headers=headers
        ) as response:
            if response.status != 200:
                response_text = await response.text()
                logger.error(
                    f"API请求失败: HTTP {response.status}, 响应: {response_text}"
                )
                response.raise_for_status()
            data = await response.json()
        # 获取消息和图片（返回与_send_image_request一致）
        if "choices" in data and data["choices"]:
            for choice in data["choices"]:
                if "message" in choice:
                    # 获取文本内容
                    content = choice["message"].get("content")
                    #if isinstance(content, str):
                    #    logger.debug("助手文本:", content)
                    # 检查是否有图片（images字段）
                    images = choice["message"].get("images")
                    if images:
                        for img in images:
                            if img.get("type") == "image_url":
                                url = img["image_url"].get("url")
                                if url and url.startswith("data:image/"):
                                    # 形如 data:image/png;base64,xxxx
                                    base64_part = url.split(",", 1)[-1]
                                    return base64.b64decode(base64_part) 
                    # 兼容部分API直接在content里返回base64
                    if isinstance(content, str) and content.startswith("![image](data:image/png;base64,"):
                        base64_part = content.split(",", 1)[-1]
                        return base64.b64decode(base64_part)
        raise Exception("未在响应中获取到图片数据")

    async def _with_retry(self, operation, *args, **kwargs):
        max_attempts = len(self.api_keys)
        if max_attempts == 0:
            return None

        for attempt in range(max_attempts):
            current_key = self._get_current_key()
            logger.info(
                f"尝试操作（密钥索引：{self.current_key_index}，次数：{attempt + 1}/{max_attempts}）"
            )
            try:
                return await operation(current_key, *args, **kwargs)
            except Exception as e:
                logger.error(f"第{attempt + 1}次尝试失败：{str(e)}")
                if attempt < max_attempts - 1:
                    self._switch_key()
                else:
                    logger.error("所有API密钥均尝试失败")
        return None

    async def terminate(self):
        if self.iwf:
            await self.iwf.terminate()
            logger.info("[ImageWorkflow] session已关闭")

