import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: 'QWEN.TextDisplay',

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const node_name = nodeData.name;
        
        // 只处理 QWEN_TextDisplay 节点
        if (node_name !== 'QWEN_TextDisplay') {
            return;
        }
        
        const onExecuted = nodeType.prototype.onExecuted;
        const onConfigure = nodeType.prototype.onConfigure;

        // populate 函数：动态创建文本显示控件
        function populate(text, name = 'text') {
            if (this.widgets) {
                const pos = this.widgets.findIndex((w) => w.name === name);
                if (pos !== -1) {
                    for (let i = pos; i < this.widgets.length; i++) {
                        this.widgets[i].onRemove?.();
                    }
                    this.widgets.length = pos;
                }
            }
            for (const list of text) {
                const w = ComfyWidgets["STRING"](this, "text", ["STRING", {multiline: true}], app).widget;
                w.inputEl.readOnly = true;
                w.inputEl.style.opacity = 0.6;
                w.value = list;
            }
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

        // When the node is executed we will be sent the input text, display this in the widget
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            populate.call(this, message.text, 'text');
        };

        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            if (this.widgets_values?.length) {
                populate.call(this, this.widgets_values, 'text');
            }
        };
    }
});
