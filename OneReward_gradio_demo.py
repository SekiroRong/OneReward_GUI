import gradio as gr
from PIL import Image, ImageDraw
import numpy as np
import torch
from diffusers.utils import load_image
from diffusers import FluxTransformer2DModel

from src.pipeline_flux_fill_with_cfg import FluxFillCFGPipeline

transformer_onereward = FluxTransformer2DModel.from_pretrained(
    "/root/autodl-tmp/OneReward",
    subfolder="flux.1-fill-dev-OneReward-transformer",
    torch_dtype=torch.bfloat16
)

pipe = FluxFillCFGPipeline.from_pretrained(
    "/root/autodl-tmp/FLUX.1-Fill-dev", 
    transformer=transformer_onereward,
    torch_dtype=torch.bfloat16).to("cuda")

# 为每种比例预设固定分辨率的蒙版
RATIO_RESOLUTIONS = {
    "1:1": (1080, 1080),      # 正方形
    "4:3": (1280, 960),       # 标准显示器比例
    "16:9": (1280, 720),      # 宽屏比例
    "21:9": (2560, 1080),     # 超宽屏比例
    "3:4": (960, 1280),       # 竖屏4:3
    "9:16": (720, 1280),      # 竖屏宽屏
    "9:21": (1080, 2560)      # 超宽竖屏
}

def resize_input_image(image, mask_width, mask_height, resolution):
    """调整输入图像大小，考虑分辨率缩放，确保不超过蒙版尺寸"""
    if image is None:
        return None
        
    # 获取原始图像尺寸和比例
    orig_width, orig_height = image.size
    orig_ratio = orig_width / orig_height
    
    # 蒙版的比例
    mask_ratio = mask_width / mask_height
    
    # 计算最大可能尺寸（刚好适应蒙版）
    if orig_ratio > mask_ratio:
        # 图像更宽，最大宽度为蒙版宽度
        max_width = mask_width
        max_height = int(mask_width / orig_ratio)
    else:
        # 图像更高，最大高度为蒙版高度
        max_height = mask_height
        max_width = int(mask_height * orig_ratio)
    
    # 确保不超过蒙版尺寸
    max_width = min(max_width, mask_width)
    max_height = min(max_height, mask_height)
    
    # 根据分辨率参数计算实际尺寸（分辨率为100时使用最大尺寸）
    scale_factor = resolution / 100
    new_width = int(max_width * scale_factor)
    new_height = int(max_height * scale_factor)
    
    # 确保至少有1x1的尺寸
    new_width = max(1, new_width)
    new_height = max(1, new_height)
    
    # 调整图像大小
    return image.resize((new_width, new_height), Image.LANCZOS)

def generate_mask_visualization_and_binary(input_image, ratio, resolution, x_position, y_position):
    """
    生成原图和固定分辨率蒙版的叠加示意图，以及二值蒙版
    返回: (可视化蒙版, 二值蒙版)
    """
    # 获取所选比例的固定分辨率蒙版尺寸
    mask_width, mask_height = RATIO_RESOLUTIONS[ratio]
    
    # 调整输入图像大小（考虑分辨率和蒙版尺寸限制）
    resized_image = resize_input_image(input_image, mask_width, mask_height, resolution)
    
    if resized_image is None:
        # 如果没有输入图像，创建一个默认的灰色图像作为示例
        default_size = min(512, mask_width, mask_height)
        resized_image = Image.new('RGB', (default_size, default_size), color='gray')
        # 根据分辨率调整默认图像
        scale_factor = resolution / 100
        new_size = int(default_size * scale_factor)
        new_size = max(1, new_size)
        resized_image = resized_image.resize((new_size, new_size), Image.LANCZOS)
    
    # 获取调整后的图像尺寸
    img_width, img_height = resized_image.size
    
    # 创建黑色背景作为可视化蒙版（固定分辨率）
    visualization_mask = Image.new('RGB', (mask_width, mask_height), color='black')
    
    # 创建二值蒙版（int8类型，0和255）
    # 初始化为255（需要扩展的区域）
    binary_mask = np.ones((mask_height, mask_width), dtype=np.uint8) * 255
    
    # 计算原图在蒙版中的位置（基于x和y位置调整）
    max_x_offset = mask_width - img_width
    max_y_offset = mask_height - img_height
    
    # 将-100到100的范围转换为0到max_offset的范围
    x = int((x_position + 100) / 200 * max_x_offset)
    y = int((y_position + 100) / 200 * max_y_offset)
    
    # 确保图像不会超出边界
    x = max(0, min(x, max_x_offset))
    y = max(0, min(y, max_y_offset))
    
    # 将调整后的图像粘贴到可视化蒙版上
    visualization_mask.paste(resized_image, (x, y))
    
    # 在二值蒙版中标记图像区域为0
    binary_mask[y:y+img_height, x:x+img_width] = 0
    
    # 将numpy数组转换为PIL图像以便处理
    binary_mask_image = Image.fromarray(binary_mask)
    
    return visualization_mask, binary_mask_image, resized_image

def expand_image(input_image, prompt, ratio, resolution, x_position, y_position):
    """使用二值蒙版进行AI扩图生成"""
    # 获取可视化蒙版、二值蒙版和调整后的图像
    visualization_mask, binary_mask, resized_image = generate_mask_visualization_and_binary(
        input_image, ratio, resolution, x_position, y_position
    )
    print(binary_mask.size, visualization_mask.size)
    
    # 获取蒙版尺寸
    mask_width, mask_height = RATIO_RESOLUTIONS[ratio]
    
    # if resized_image is None:
    #     # 如果没有输入图像，创建一个默认图像
    #     default_size = min(512, mask_width, mask_height)
    #     resized_image = Image.new('RGB', (default_size, default_size), color='lightblue')
    #     draw = ImageDraw.Draw(resized_image)
    #     draw.text((10, resized_image.height//2), "No Image", fill='black')
    #     # 根据分辨率调整默认图像
    #     scale_factor = resolution / 100
    #     new_size = int(default_size * scale_factor)
    #     new_size = max(1, new_size)
    #     resized_image = resized_image.resize((new_size, new_size), Image.LANCZOS)
    
    # 将二值蒙版转换为numpy数组以便处理
    binary_mask_np = np.array(binary_mask)

    image = pipe(
        prompt=prompt,
        negative_prompt="",
        image=visualization_mask,
        mask_image=binary_mask,
        height=visualization_mask.height,
        width=visualization_mask.width,
        guidance_scale=1.0,
        true_cfg=4.0,
        num_inference_steps=50,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    
    # # 模拟AI扩图 - 在实际应用中这里会使用binary_mask_np作为输入
    # # 来指导AI生成需要扩展的区域
    # result = Image.new('RGB', (mask_width, mask_height))
    
    # # 创建渐变背景（模拟AI生成的内容），但考虑二值蒙版
    # for y_coord in range(mask_height):
    #     for x_coord in range(mask_width):
    #         # 根据二值蒙版决定颜色 - 0区域保留原图，255区域生成新内容
    #         if binary_mask_np[y_coord, x_coord] == 0:
    #             # 这里会被原图覆盖，所以暂时用黑色填充
    #             r, g, b = 0, 0, 0
    #         else:
    #             # 生成渐变背景（模拟AI生成的内容）
    #             r = int(100 + x_coord * 155 / mask_width)
    #             g = int(100 + y_coord * 155 / mask_height)
    #             b = 255
    #         result.putpixel((x_coord, y_coord), (r, g, b))
    
    # # 计算原图在结果中的位置
    # img_width, img_height = resized_image.size
    # max_x_offset = mask_width - img_width
    # max_y_offset = mask_height - img_height
    # x = int((x_position + 100) / 200 * max_x_offset)
    # y = int((y_position + 100) / 200 * max_y_offset)
    # x = max(0, min(x, max_x_offset))
    # y = max(0, min(y, max_y_offset))
    
    # # 将调整后的原图粘贴到指定位置
    # result.paste(resized_image, (x, y))
    
    return image

def update_masks(input_image, ratio, resolution, x_position, y_position):
    """更新蒙版预览图"""
    vis_mask, bin_mask, _ = generate_mask_visualization_and_binary(
        input_image, ratio, resolution, x_position, y_position
    )
    return vis_mask, bin_mask

# 当比例改变时，显示当前选中的蒙版分辨率
def display_mask_resolution(ratio):
    width, height = RATIO_RESOLUTIONS[ratio]
    return f"蒙版分辨率: {width} × {height}"

with gr.Blocks(title="AI图像扩展工具") as demo:
    gr.Markdown("## AI图像扩展工具")
    
    with gr.Row():
        # 左侧输入区域
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="输入图像")
            prompt = gr.Textbox(label="描述文本", placeholder="输入对扩展区域的描述...")
            
            # 输入图片分辨率控制
            input_resolution = gr.Slider(
                minimum=10,
                maximum=100,
                value=70,
                step=5,
                label="输入图片大小 (%)"
            )
            
            # X和Y方向位置控制滑块
            with gr.Row():
                x_position = gr.Slider(
                    minimum=-100,
                    maximum=100,
                    value=0,
                    step=5,
                    label="X轴位置"
                )
                y_position = gr.Slider(
                    minimum=-100,
                    maximum=100,
                    value=0,
                    step=5,
                    label="Y轴位置"
                )
            
            generate_btn = gr.Button("生成扩展图像", variant="primary")
        
        # 右侧参数和预览区域
        with gr.Column(scale=1):
            ratio = gr.Dropdown(
                choices=["1:1", "4:3", "16:9", "21:9", "3:4", "9:16", "9:21"],
                label="扩展比例",
                value="16:9"
            )
            
            # 显示当前选中的蒙版分辨率
            mask_res_display = gr.Textbox(
                label="蒙版信息",
                value=display_mask_resolution("1:1"),
                interactive=False
            )
            
            # 扩展区域预览
            mask_visualization = gr.Image(label="扩展区域预览")
            
            # 二值蒙版（设置为不可见）
            binary_mask = gr.Image(
                label="二值蒙版 (0=原图区域, 255=扩展区域)",
                visible=False
            )
            
            # 最终输出
            output_image = gr.Image(label="扩展结果")
    
    # 设置更新事件 - 当任何参数变化时更新预览
    ratio.change(
        fn=update_masks,
        inputs=[input_image, ratio, input_resolution, x_position, y_position],
        outputs=[mask_visualization, binary_mask]
    ).then(
        fn=display_mask_resolution,
        inputs=[ratio],
        outputs=[mask_res_display]
    )
    
    input_resolution.change(
        fn=update_masks,
        inputs=[input_image, ratio, input_resolution, x_position, y_position],
        outputs=[mask_visualization, binary_mask]
    )
    
    x_position.change(
        fn=update_masks,
        inputs=[input_image, ratio, input_resolution, x_position, y_position],
        outputs=[mask_visualization, binary_mask]
    )
    
    y_position.change(
        fn=update_masks,
        inputs=[input_image, ratio, input_resolution, x_position, y_position],
        outputs=[mask_visualization, binary_mask]
    )
    
    input_image.change(
        fn=update_masks,
        inputs=[input_image, ratio, input_resolution, x_position, y_position],
        outputs=[mask_visualization, binary_mask]
    )
    
    # 生成按钮点击事件
    generate_btn.click(
        fn=expand_image,
        inputs=[input_image, prompt, ratio, input_resolution, x_position, y_position],
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006)
