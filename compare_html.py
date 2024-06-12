import os

def generate_html(folder_path1, folder_path2):
    rows1 = sorted(os.listdir(folder_path1))
    rows2 = sorted(os.listdir(folder_path2))
    html = "<html><head><meta charset='UTF-8'></head><body><table style='width:100%;border-collapse: collapse;'>"

    # 添加表头
    html += "<thead><tr><th></th>"  # 空单元格，用于行名和列名的交叉点
    column_names = ["RandomSeed_1", "RandomSeed_1", "RandomSeed_2", "RandomSeed_3"]
    for col_name in column_names:
        html += f"<th style='border: 1px solid black;padding: 10px;'>{col_name}</th>"
    html += "</tr></thead>"

    # 添加表格内容
    for i in range(max(len(rows1), len(rows2))):
        for folder_path, rows in [(folder_path1, rows1), (folder_path2, rows2)]:
            if i < len(rows):
                row = rows[i]
                row_path = os.path.join(folder_path, row)
                if os.path.isdir(row_path):
                    html += "<tr>"

                    # 添加行名
                    html += f"<th style='border: 1px solid black;padding: 10px;'>{os.path.join(folder_path, row)}</th>"

                    images = sorted(os.listdir(row_path))[:4]
                    for img in images:
                        img_path = os.path.join(folder_path, row, img)
                        html += f"<td style='border: 1px solid black;padding: 10px;'><img src='{img_path}' alt='{img}' style='width:100%'></td>"
                    html += "</tr>"

    html += "</table></body></html>"

    with open(f"{'-'.join(folder_path1.split('/'))}-vs-{'-'.join(folder_path2.split('/'))}.html", "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    folder_path1 = "results/cascade_inference/idm-vton-dc/test_ali"  # 将此更改为实际文件夹路径
    folder_path2 = "results/cascade_inference/idm-vton-dc-lower-body-lr-1e-5-bsz-32-bf16-resume-3000/checkpoint-2000/test_ali"  # 将此更改为实际文件夹路径
    generate_html(folder_path1, folder_path2)