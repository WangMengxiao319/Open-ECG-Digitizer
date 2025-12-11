

import os
from pdf2image import convert_from_path

# ==== 配置路径 ====
input_dir = r"../急诊心电图-常州二院/pdf"     # PDF 文件所在文件夹
output_dir = r"../急诊心电图-常州二院/png"    # PNG 输出文件夹
poppler_path = r"D:\poppler-25.12.0\Library\bin"   # 你的 poppler 路径（Windows 必填）

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 获取 PDF 文件列表
pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]

for pdf_file in pdf_files:
    pdf_path = os.path.join(input_dir, pdf_file)
    png_name = os.path.splitext(pdf_file)[0] + ".png"
    png_path = os.path.join(output_dir, png_name)

    # 转换 PDF → PNG（每个 PDF 只有一页）
    images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
    
    # # 裁剪，高度方向从1050裁剪到4200
    # images[0] = images[0].crop((0, 980, images[0].width, 4150))

    # 取第一页并保存 PNG
    images[0].save(png_path, "PNG")

    print(f"转换完成: {pdf_file} → {png_name}")

print("全部转换完成！")
