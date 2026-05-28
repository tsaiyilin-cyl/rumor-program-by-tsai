import shutil
from pathlib import Path

def step(path="data_DeepSeek-V3.2"):
    '''
    将所有的opinion_trend合到一个文件夹，方便查看
    '''
    # 基础路径
    base_dir = Path(__file__).parent/path          # A 文件夹
    logs_dir = base_dir / "logs"               # logs 文件夹
    output_dir = base_dir / "merged_opinion_trends"  # 汇总输出文件夹
    output_dir.mkdir(exist_ok=True)

    # 遍历 logs 下的所有子文件夹
    for subdir in logs_dir.iterdir():
        if not subdir.is_dir():
            continue

        # 提取 topic 序号（例如 "topic0"）
        topic = subdir.name.split('_')[0]       # 假设文件夹名以 topicX 开头

        # 图片源路径
        src_img = subdir / "opinion_trend.png"
        if not src_img.exists():
            print(f"警告: {src_img} 不存在，跳过")
            continue

        # 创建该 topic 的目标子文件夹
        topic_dir = output_dir / topic
        topic_dir.mkdir(exist_ok=True)

        # 目标图片名：使用原始文件夹名 + ".png"，避免同一 topic 下重名
        dst_img = topic_dir / f"{subdir.name}.png"

        # 复制文件（如果目标已存在，根据需求选择是否覆盖）
        shutil.copy2(src_img, dst_img)   # copy2 保留元数据
        print(f"已复制: {src_img} -> {dst_img}")
def main():
    step()
if __name__ == "__main__":
    step()