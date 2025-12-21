import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// QWEN 文本展示节点的前端扩展
app.registerExtension({
    name: "QWEN.TextDisplay",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const node_name = nodeData.name;
        
        // 只处理 QWEN_TextDisplay 节点
        if (node_name !== 'QWEN_TextDisplay') {
            return;
        }
        
        // 保存原始方法
        const onExecuted = nodeType.prototype.onExecuted;
        const onConfigure = nodeType.prototype.onConfigure;
        
        // populate 函数：动态创建文本显示控件
        function populate(text) {
            // 清除已有的显示控件
            if (this.widgets) {
                const pos = this.widgets.findIndex((w) => w.name === "text");
                if (pos !== -1) {
                    // 移除旧的显示控件
                    for (let i = pos; i < this.widgets.length; i++) {
                        this.widgets[i].onRemove?.();
                    }
                    this.widgets.length = pos;
                }
            }
            
            // 为每个文本项创建一个只读文本框
            for (const list of text) {
                const w = ComfyWidgets["STRING"](
                    this, 
                    "text", 
                    ["STRING", { multiline: true }], 
                    app
                ).widget;
                
                w.inputEl.readOnly = true;
                w.inputEl.style.opacity = 0.6;
                w.value = list;
            }
            
            // 调整节点大小
            requestAnimationFrame(() => {
                const sz = this.computeSize();
                if (sz[0] < this.size[0]) {
                    sz[0] = this.size[0];
                }
                if (sz[1] < this.size[1]) {
                    sz[1] = this.size[1];
                }
                this.onResize?.(sz);
                app.graph.setDirtyCanvas(true, false);
            });
        }
        
        // 监听节点执行完成事件
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            
            // 如果后端返回了 text 数据，显示它
            if (message?.text) {
                populate.call(this, message.text);
            }
        };
        
        // 加载工作流时恢复显示
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            
            if (this.widgets_values?.length) {
                populate.call(this, this.widgets_values);
            }
        };
    }
});
