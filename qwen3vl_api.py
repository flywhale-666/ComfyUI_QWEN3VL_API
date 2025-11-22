"""
QWEN3-VL API Node for ComfyUI
Based on Alibaba Cloud DashScope API
"""

import os
import base64
import torch
import random
import glob
from io import BytesIO
from PIL import Image


def log(message, message_type='info'):
    """日志输出函数"""
    name = 'QWEN3VL_API'
    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    else:
        message = '\033[1;33m' + message + '\033[m'
    print(f"# 🤖 {name} -> {message}")


def tensor2pil(t_image):
    """将 Tensor 转换为 PIL Image"""
    import numpy as np
    return Image.fromarray(
        (t_image.cpu().numpy().squeeze() * 255).astype('uint8')
    )


def get_api_key():
    """从 api_key.ini 文件读取 API Key"""
    api_key_file = os.path.join(
        os.path.dirname(os.path.normpath(__file__)), 
        "api_key.ini"
    )
    
    api_key = ''
    try:
        with open(api_key_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if line.startswith('DASHSCOPE_API_KEY='):
                    api_key = line.split('=', 1)[1].strip()
                    break
    except FileNotFoundError:
        log(f'❌ 配置文件不存在: {api_key_file}', message_type='error')
        log(f'请创建 api_key.ini 文件并填写 DASHSCOPE_API_KEY', message_type='warning')
        return ''
    except Exception as e:
        log(f'❌ 读取配置文件失败: {repr(e)}', message_type='error')
        return ''
    
    # 移除可能的引号
    remove_chars = ['"', "'", '"', '"', ''', ''']
    for char in remove_chars:
        api_key = api_key.replace(char, '')
    
    if len(api_key) < 10:
        log(f'❌ API Key 无效，请检查 {api_key_file}', message_type='error')
        return ''
    
    return api_key


class QWEN3VL_Image:
    """QWEN3-VL 图像理解节点"""
    
    def __init__(self):
        self.NODE_NAME = 'QWEN3VL_Image'
    
    @classmethod
    def INPUT_TYPES(cls):
        model_list = [
            "qwen3-vl-flash",
            "qwen3-vl-flash-2025-10-15",
            "qwen3-vl-plus",
            "qwen3-vl-plus-2025-09-23",
            "qwen-vl-max",
        ]
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (model_list,),
                "user_prompt": ("STRING", {
                    "default": "请描述这张图片", 
                    "multiline": True
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "qwen3vl_image"
    CATEGORY = '🤖QWEN3VL_API'
    
    @classmethod
    def IS_CHANGED(cls, image, model, user_prompt, seed):
        """返回唯一值，只有当这些参数变化时才重新执行"""
        # 使用 seed 和其他参数来决定是否缓存
        # 返回 seed 即可，seed 变化则重新执行
        return seed
    
    def qwen3vl_image(self, image, model, user_prompt, seed):
        """调用 QWEN3-VL API 进行图像理解"""
        from openai import OpenAI
        
        # 输出调试信息，查看 seed 的类型和值
        log(f"接收到 seed: {seed}, 类型: {type(seed)}")
        
        # 确保 seed 是整数类型，处理可能的浮点数或其他类型
        try:
            seed = int(float(seed))
            # 限制 seed 在 32 位有符号整数范围内 (-2147483648 到 2147483647)
            # API 可能只支持标准的 32 位整数
            if seed > 2147483647:
                seed = seed % 2147483647
            elif seed < 0:
                seed = abs(seed) % 2147483647
        except (ValueError, TypeError) as e:
            log(f"警告: seed 类型转换失败 {e}，使用默认值 0", message_type='warning')
            seed = 0
        
        log(f"转换后 seed: {seed}, 类型: {type(seed)}")
        
        # 获取 API Key
        api_key = get_api_key()
        if not api_key:
            return ("❌ 未配置 API Key，请检查 api_key.ini 文件",)
        
        # 初始化客户端
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        # 将 tensor 转换为 PIL Image
        img = tensor2pil(image).convert('RGB')
        
        # 将图片转换为 base64
        img_data = BytesIO()
        img.save(img_data, format="JPEG")
        img_url = base64.b64encode(img_data.getvalue()).decode("utf-8")
        
        # 构建消息
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_url}"}
                },
                {
                    "type": "text",
                    "text": user_prompt
                }
            ]
        }]
        
        try:
            # 调用 API
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                seed=seed
            )
            
            ret_message = response.choices[0].message.content
            log(f"{self.NODE_NAME} 响应 (seed={seed}): {ret_message}")
            
            return (ret_message,)
            
        except Exception as e:
            error_msg = f"❌ API 调用失败: {repr(e)}"
            log(error_msg, message_type='error')
            return (error_msg,)


class LoadImageFromFolder:
    """从文件夹加载图像节点"""
    
    def __init__(self):
        self.NODE_NAME = 'LoadImageFromFolder'
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "", 
                    "multiline": False
                }),
                "image_limit": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1
                }),
                "start_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1
                }),
                "sort_method": (["None", "Alphabetical (ASC)", "Alphabetical (DESC)", 
                                "Numerical (ASC)", "Numerical (DESC)", 
                                "Datetime (ASC)", "Datetime (DESC)"],),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "file_paths")
    FUNCTION = "load_images"
    CATEGORY = '🤖QWEN3VL_API'
    OUTPUT_IS_LIST = (True, True)
    
    def load_images(self, folder_path, image_limit, start_index, sort_method):
        """从文件夹加载图像文件"""
        
        if not os.path.isdir(folder_path):
            error_msg = f"❌ 文件夹不存在: {folder_path}"
            log(error_msg, message_type='error')
            return ([], [])
        
        # 支持的图像扩展名
        image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp', 'tiff', 'tif']
        
        # 查找所有图像文件
        image_files = []
        for ext in image_extensions:
            pattern = os.path.join(folder_path, f"*.{ext}")
            image_files.extend(glob.glob(pattern))
        
        if not image_files:
            log(f"❌ 在文件夹 {folder_path} 中未找到图像文件", message_type='warning')
            return ([], [])
        
        # 排序
        if sort_method == "Alphabetical (ASC)":
            image_files.sort(key=lambda x: os.path.basename(x).lower())
        elif sort_method == "Alphabetical (DESC)":
            image_files.sort(key=lambda x: os.path.basename(x).lower(), reverse=True)
        elif sort_method == "Numerical (ASC)":
            import re
            def numerical_sort_key(path):
                name = os.path.basename(path)
                numbers = re.findall(r'\d+', name)
                return [int(n) for n in numbers] if numbers else [0]
            image_files.sort(key=numerical_sort_key)
        elif sort_method == "Numerical (DESC)":
            import re
            def numerical_sort_key(path):
                name = os.path.basename(path)
                numbers = re.findall(r'\d+', name)
                return [int(n) for n in numbers] if numbers else [0]
            image_files.sort(key=numerical_sort_key, reverse=True)
        elif sort_method == "Datetime (ASC)":
            image_files.sort(key=lambda x: os.path.getmtime(x))
        elif sort_method == "Datetime (DESC)":
            image_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        # sort_method == "None" 时保持原顺序
        
        # 应用起始索引
        if start_index >= len(image_files):
            log(f"⚠️ 起始索引 {start_index} 超出范围，共有 {len(image_files)} 个图像", message_type='warning')
            return ([], [])
        
        image_files = image_files[start_index:]
        
        # 应用加载上限
        if image_limit > 0:
            image_files = image_files[:image_limit]
        
        log(f"加载了 {len(image_files)} 个图像文件")
        
        # 加载图像并转换为 tensor
        images = []
        file_paths = []
        
        for image_path in image_files:
            # 检查文件是否存在
            if os.path.isfile(image_path):
                try:
                    img = Image.open(image_path).convert('RGB')
                    # 转换为 ComfyUI 的 tensor 格式
                    import numpy as np
                    img_array = np.array(img).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_array)[None,]
                    
                    images.append(img_tensor)
                    file_paths.append(image_path)
                    log(f"✓ {os.path.basename(image_path)}")
                except Exception as e:
                    log(f"⚠️ 加载图像失败 {os.path.basename(image_path)}: {repr(e)}", message_type='warning')
            else:
                log(f"⚠️ 跳过不存在的文件: {image_path}", message_type='warning')
        
        if not images:
            log("❌ 没有有效的图像文件", message_type='error')
            return ([], [])
        
        log(f"成功加载 {len(images)} 个图像", message_type='finish')
        return (images, file_paths)


class LoadVideoFromFolder:
    """从文件夹加载视频节点"""
    
    def __init__(self):
        self.NODE_NAME = 'LoadVideoFromFolder'
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "", 
                    "multiline": False
                }),
                "video_limit": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1
                }),
                "start_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1
                }),
                "sort_method": (["None", "Alphabetical (ASC)", "Alphabetical (DESC)", 
                                "Numerical (ASC)", "Numerical (DESC)", 
                                "Datetime (ASC)", "Datetime (DESC)"],),
            },
        }
    
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("videos", "file_paths")
    FUNCTION = "load_videos"
    CATEGORY = '🤖QWEN3VL_API'
    OUTPUT_IS_LIST = (True, True)
    
    def load_videos(self, folder_path, video_limit, start_index, sort_method):
        """从文件夹加载视频文件"""
        
        if not os.path.isdir(folder_path):
            error_msg = f"❌ 文件夹不存在: {folder_path}"
            log(error_msg, message_type='error')
            return ([], [])
        
        # 支持的视频扩展名
        video_extensions = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv', 'mpeg', 'mpg']
        
        # 查找所有视频文件
        video_files = []
        for ext in video_extensions:
            pattern = os.path.join(folder_path, f"*.{ext}")
            video_files.extend(glob.glob(pattern))
        
        if not video_files:
            log(f"❌ 在文件夹 {folder_path} 中未找到视频文件", message_type='warning')
            return ([], [])
        
        # 排序
        if sort_method == "Alphabetical (ASC)":
            video_files.sort(key=lambda x: os.path.basename(x).lower())
        elif sort_method == "Alphabetical (DESC)":
            video_files.sort(key=lambda x: os.path.basename(x).lower(), reverse=True)
        elif sort_method == "Numerical (ASC)":
            import re
            def numerical_sort_key(path):
                name = os.path.basename(path)
                numbers = re.findall(r'\d+', name)
                return [int(n) for n in numbers] if numbers else [0]
            video_files.sort(key=numerical_sort_key)
        elif sort_method == "Numerical (DESC)":
            import re
            def numerical_sort_key(path):
                name = os.path.basename(path)
                numbers = re.findall(r'\d+', name)
                return [int(n) for n in numbers] if numbers else [0]
            video_files.sort(key=numerical_sort_key, reverse=True)
        elif sort_method == "Datetime (ASC)":
            video_files.sort(key=lambda x: os.path.getmtime(x))
        elif sort_method == "Datetime (DESC)":
            video_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        # sort_method == "None" 时保持原顺序
        
        # 应用起始索引
        if start_index >= len(video_files):
            log(f"⚠️ 起始索引 {start_index} 超出范围，共有 {len(video_files)} 个视频", message_type='warning')
            return ([], [])
        
        video_files = video_files[start_index:]
        
        # 应用加载上限
        if video_limit > 0:
            video_files = video_files[:video_limit]
        
        log(f"加载了 {len(video_files)} 个视频文件")
        
        # 返回视频路径列表（ComfyUI的VIDEO类型通常是文件路径）
        videos = []
        file_paths = []
        
        for video_path in video_files:
            # 检查文件是否存在
            if os.path.isfile(video_path):
                videos.append(video_path)
                file_paths.append(video_path)
                log(f"✓ {os.path.basename(video_path)}")
            else:
                log(f"⚠️ 跳过不存在的文件: {video_path}", message_type='warning')
        
        if not videos:
            log("❌ 没有有效的视频文件", message_type='error')
            return ([], [])
        
        log(f"成功加载 {len(videos)} 个视频", message_type='finish')
        return (videos, file_paths)


class QWEN3VL_Video:
    """QWEN3-VL 视频理解节点（支持路径输入或VIDEO输入）"""
    
    def __init__(self):
        self.NODE_NAME = 'QWEN3VL_Video'
    
    @classmethod
    def INPUT_TYPES(cls):
        model_list = [
            "qwen3-vl-flash",
            "qwen3-vl-flash-2025-10-15",
            "qwen3-vl-plus",
            "qwen3-vl-plus-2025-09-23",
            "qwen-vl-max",
        ]
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "", 
                    "multiline": False
                }),
                "model": (model_list,),
                "user_prompt": ("STRING", {
                    "default": "请描述这个视频的内容", 
                    "multiline": True
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff
                }),
            },
            "optional": {
                "video": ("VIDEO",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "qwen3vl_video"
    CATEGORY = '🤖QWEN3VL_API'
    
    @classmethod
    def IS_CHANGED(cls, model, user_prompt, seed, video_path="", video=None):
        return seed
    
    def qwen3vl_video(self, model, user_prompt, seed, video_path="", video=None):
        """调用 QWEN3-VL API 进行视频理解"""
        from openai import OpenAI
        import mimetypes
        
        # 处理 seed
        try:
            seed = int(float(seed))
            if seed > 2147483647:
                seed = seed % 2147483647
            elif seed < 0:
                seed = abs(seed) % 2147483647
        except (ValueError, TypeError) as e:
            log(f"警告: seed 类型转换失败 {e}，使用默认值 0", message_type='warning')
            seed = 0
        
        # 决定视频路径：优先使用 VIDEO 输入，其次使用 video_path
        final_path = None
        
        # 优先处理 VIDEO 输入
        if video is not None:
            # 尝试多种可能的格式
            if isinstance(video, dict):
                # 尝试常见的键名
                for key in ['filename', 'path', 'file', 'video_path', 'filepath']:
                    if key in video:
                        final_path = video[key]
                        break
            elif isinstance(video, str):
                final_path = video
            elif isinstance(video, (list, tuple)) and len(video) > 0:
                # 如果是列表或元组，取第一个元素
                first_item = video[0]
                if isinstance(first_item, dict):
                    for key in ['filename', 'path', 'file', 'video_path', 'filepath']:
                        if key in first_item:
                            final_path = first_item[key]
                            break
                elif isinstance(first_item, str):
                    final_path = first_item
            else:
                # 处理 ComfyUI 的 VideoFromFile 对象或其他对象
                # 方法1: 尝试从 __dict__ 获取
                if hasattr(video, '__dict__'):
                    obj_dict = video.__dict__
                    # 查找包含文件路径的键（支持私有属性如 _VideoFromFile__file）
                    for key, value in obj_dict.items():
                        if isinstance(value, str) and ('file' in key.lower() or 'path' in key.lower()):
                            if os.path.isfile(value):
                                final_path = value
                                break
                
                # 方法2: 尝试常见的属性名
                if final_path is None:
                    for attr in ['file', 'filename', 'path', 'filepath', 'video_path']:
                        if hasattr(video, attr):
                            try:
                                value = getattr(video, attr)
                                if isinstance(value, str) and os.path.isfile(value):
                                    final_path = value
                                    break
                            except Exception:
                                pass
        
        # 如果没有 VIDEO 输入，使用 video_path
        if final_path is None and video_path and video_path.strip():
            final_path = video_path.strip()
        
        # 检查是否提供了视频
        if not final_path:
            error_msg = "❌ 未提供视频，请使用 VIDEO 输入或填写 video_path"
            log(error_msg, message_type='error')
            return (error_msg,)
        
        # 检查文件是否存在
        if not os.path.isfile(final_path):
            error_msg = f"❌ 视频文件不存在: {final_path}"
            log(error_msg, message_type='error')
            return (error_msg,)
        
        log(f"使用视频文件: {final_path}")
        
        api_key = get_api_key()
        if not api_key:
            return ("❌ 未配置 API Key，请检查 api_key.ini 文件",)
        
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        mime, _ = mimetypes.guess_type(final_path)
        mime = mime or "video/mp4"
        
        try:
            with open(final_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            video_url = f"data:{mime};base64,{b64}"
        except Exception as e:
            return (f"❌ 读取视频文件失败: {repr(e)}",)
        
        # 构建消息（按照官方示例格式）
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": video_url}
                },
                {
                    "type": "text",
                    "text": user_prompt
                }
            ]
        }]
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                seed=seed
            )
            
            raw = response.choices[0].message.content or ""
            # 清理响应中的特殊标记
            ret = raw.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "").strip()
            log(f"{self.NODE_NAME} 响应 (seed={seed}): {ret}")
            
            return (ret,)
            
        except Exception as e:
            error_msg = f"❌ API 调用失败: {repr(e)}"
            log(error_msg, message_type='error')
            return (error_msg,)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "QWEN3VL_Image": QWEN3VL_Image,
    "QWEN3VL_Video": QWEN3VL_Video,
    "LoadImageFromFolder": LoadImageFromFolder,
    "LoadVideoFromFolder": LoadVideoFromFolder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QWEN3VL_Image": "QWEN3-VL 图像理解",
    "QWEN3VL_Video": "QWEN3-VL 视频理解",
    "LoadImageFromFolder": "QWEN3-VL 加载图像(文件夹)",
    "LoadVideoFromFolder": "QWEN3-VL 加载视频(文件夹)",
}
