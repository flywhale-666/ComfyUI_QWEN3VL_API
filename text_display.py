"""
QWEN 文本展示节点
用于在节点上显示文本内容，支持批次输入输出
"""


class QWEN_TextDisplay:
    """文本展示节点 - 在节点上显示字符串文本"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "forceInput": True
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    OUTPUT_NODE = True
    FUNCTION = "display_text"
    CATEGORY = '🤖QWEN3VL_API'
    
    def display_text(self, text, unique_id=None, extra_pnginfo=None):
        """显示文本并传递"""
        
        # 处理输入 - text 是列表
        display_values = []
        
        for item in text:
            if isinstance(item, str):
                display_values.append(item)
            else:
                display_values.append(str(item))
        
        # 返回 UI 显示数据和结果
        return {
            "ui": {"text": display_values}, 
            "result": (display_values,)
        }


# 节点映射
NODE_CLASS_MAPPINGS = {
    "QWEN_TextDisplay": QWEN_TextDisplay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QWEN_TextDisplay": "QWEN 文本展示",
}
